// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include "mpi.h"
#include <chrono>
#include <iomanip>

#ifdef STANDALONE_BENCHMARK

MPI_Comm comm;
int comm_rank;
int comm_size;

#else

#include "../common/dr_bench.hpp"

#endif

namespace ShallowWater {

using T = double;

using Array = dr::mhp::distributed_mdarray<T, 2>;

// gravitational acceleration
constexpr double g = 9.81;
// Coriolis parameter
constexpr double f = 10.0;
// Length scale of geostrophic gyre
constexpr double sigma = 0.4;

void printArray(Array &arr, std::string msg) {
  std::cout << msg << ":\n";
  for (auto segment : dr::ranges::segments(arr)) {
    if (dr::ranges::rank(segment) == std::size_t(comm_rank)) {
      auto origin = segment.origin();
      auto s = segment.mdspan();
      for (std::size_t i = 0; i < s.extent(0); i++) {
        std::size_t global_i = i + origin[0];
          printf("%3zu: ", global_i);
          for (std::size_t j = 0; j < s.extent(1); j++) {
            printf("%9.6f ", s(i,j));
          }
          std::cout << "\n";
      }
      std::cout << "\n" << std::flush;
    }
  }
}

std::size_t shape(const Array &arr, std::size_t dim) {
  return arr.mdspan().extent(dim);
}

// Arakava C grid object
//
// T points at cell centers
// U points at center of x edges
// V points at center of y edges
// F points at vertices
//
//   |       |       |       |       |
//   f---v---f---v---f---v---f---v---f-
//   |       |       |       |       |
//   u   t   u   t   u   t   u   t   u
//   |       |       |       |       |
//   f---v---f---v---f---v---f---v---f-
struct ArakawaCGrid {
  double xmin, xmax;     // x limits in physical coordinates (U point min/max)
  double ymin, ymax;     // y limits in physical coordinates (V point min/max)
  std::size_t nx, ny;    // number of cells (T points)
  double lx, ly;         // grid size in physical coordinates
  double dx, dy;         // cell size in physical coordinates
  double dx_inv, dy_inv; // reciprocial dx and dy

  ArakawaCGrid(double _xmin, double _xmax, double _ymin, double _ymax, std::size_t _nx, std::size_t _ny) : xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax), nx(_nx), ny(_ny), lx(_xmax - _xmin), ly(_ymax - _ymin){
    dx = lx / nx;
    dy = ly / ny;
    dx_inv = 1.0 / dx;
    dy_inv = 1.0 / dy;
  };
};

// FIXME
// Get number of read/write bytes and flops for a single time step
// These numbers correspond to the fused kernel version
void calculate_complexity(std::size_t nx, std::size_t ny, std::size_t &nread,
                              std::size_t &nwrite, std::size_t &nflop) {
  // stage1: 2+2+3 = 7
  // stage2: 3+3+4 = 10
  // stage3: 3+3+4 = 10
  nread = (27 * nx * ny) * sizeof(T);
  // stage1: 3
  // stage2: 3
  // stage3: 3
  nwrite = (9 * nx * ny) * sizeof(T);
  // stage1: 3+3+6 = 12
  // stage2: 6+6+9 = 21
  // stage3: 6+6+9 = 21
  nflop = 54 * nx * ny;
}

double exact_elev(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for elevation field.
   *
   * Returns time-dependent elevation of a 2D stationary gyre.
   */
  double amp = 0.02;
  return amp*exp(-(x*x + y*y)/sigma/sigma);
}

double exact_u(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for x velocity field.
   *
   * Returns time-dependent velocity of a 2D stationary gyre.
   */
  double elev = exact_elev(x, y, t, lx, ly);
  return g/f*2*y/sigma/sigma*elev;
}

double exact_v(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for y velocity field.
   *
   * Returns time-dependent velocity of a 2D stationary gyre.
   */
  double elev = exact_elev(x, y, t, lx, ly);
  return -g/f*2*x/sigma/sigma*elev;
}

double bathymetry(double x, double y, double lx, double ly) {
  /**
   * Bathymetry, i.e. water depth at rest.
   *
   */
  // return 1.0;
  return 10*exact_elev(x, y, 0.0, lx, ly) + 1.0; // FIXME nontrivial for testing
}

double initial_elev(double x, double y, double lx, double ly) {
  return exact_elev(x, y, 0.0, lx, ly);
}

double initial_u(double x, double y, double lx, double ly) {
  return exact_u(x, y, 0.0, lx, ly);
}

double initial_v(double x, double y, double lx, double ly) {
  return exact_v(x, y, 0.0, lx, ly);
}

void set_field(Array &arr,
               std::function<double(double, double, double, double)> func,
               ArakawaCGrid &grid,
               double x_offset = 0.0, double y_offset = 0.0, int row_offset = 0) {
  /**
   * Assign Array values based on coordinate-dependent function `func`.
   * 
   */
  for (auto segment : dr::ranges::segments(arr)) {
    if (dr::ranges::rank(segment) == std::size_t(comm_rank)) {
      auto origin = segment.origin();
      auto s = segment.mdspan();

      for (std::size_t i = 0; i < s.extent(0); i++) {
        std::size_t global_i = i + origin[0];
        if (global_i >= row_offset) {
          for (std::size_t j = 0; j < s.extent(1); j++) {
            std::size_t global_j = j + origin[1];
            T x = grid.xmin + x_offset + (global_i - row_offset) * grid.dx;
            T y = grid.ymin + y_offset + global_j * grid.dy;
            s(i, j) = func(x, y, grid.lx, grid.ly);
          }
        }
      }
    }
  }
}

void rhs(Array &u, Array &v, Array &e, Array &hu, Array &hv,
         Array &dudy, Array &dvdx,
         Array &dudx, Array &dvdy,
         Array &dudt, Array &dvdt, Array &dedt,
         Array &h, double g, double f, double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate right hand side of the equations
   */
  auto rhs_dudy = [dy_inv](auto args) {
    auto [u, out] = args;
    out(0, 0) = (u(0, 0) - u(0, -1)) * dy_inv;
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{shape(dudy, 0), shape(dudy, 1)-1};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto dudy_view = dr::mhp::views::submdspan(dudy.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dudy, u_view, dudy_view);
  }

  auto rhs_dvdy = [dy_inv](auto args) {
    auto [v, out] = args;
    out(0, 0) = (v(0, 0) - v(0, -1)) * dy_inv;
  };
  {
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{shape(dvdy, 0), shape(dvdy, 1)};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dvdy_view = dr::mhp::views::submdspan(dvdy.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dvdy, v_view, dvdy_view);
  }

  dr::mhp::halo(u).exchange_finalize();
  auto rhs_dudx = [dx_inv](auto args) {
    auto [u, out] = args;
    out(0, 0) = (u(0, 0) - u(-1, 0)) * dx_inv;
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(dudx, 0), shape(dudx, 1)};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto dudx_view = dr::mhp::views::submdspan(dudx.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dudx, u_view, dudx_view);
  }
  dr::mhp::halo(dudx).exchange_begin();

  dr::mhp::halo(v).exchange_finalize();
  auto rhs_dvdx = [dx_inv](auto args) {
    auto [v, out] = args;
    out(0, 0) = (v(1, 0) - v(0, 0)) * dx_inv;
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(dvdx, 0)-1, shape(dvdx, 1)};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dvdx_view = dr::mhp::views::submdspan(dvdx.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dvdx, v_view, dvdx_view);
  }
  dr::mhp::halo(dvdx).exchange_begin();

  dr::mhp::halo(e).exchange_finalize();
  dr::mhp::halo(dudx).exchange_finalize();
  auto rhs_dudt = [dt, g, f, dx_inv](auto tuple) {
    auto [e, u, v, dudx, dudy, out] = tuple;
    auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
    auto v_at_u = 0.25 * (v(0, 0) + v(1, 0) + v(0, 1) + v(1, 1));
    auto dudx_up = u(0, 0) > 0 ? dudx(0, 0) : dudx(1, 0);
    auto dudy_up = v_at_u > 0 ? dudy(0, 0) : dudy(0, 1);
    out(0, 0) = dt * (-g * dedx + f * v_at_u - u(0, 0) * dudx_up - v_at_u * dudy_up);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(e, 0) - 1, shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dudx_view = dr::mhp::views::submdspan(dudx.view(), start, end);
    auto dudy_view = dr::mhp::views::submdspan(dudy.view(), start, end);
    auto dudt_view = dr::mhp::views::submdspan(dudt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dudt, e_view, u_view, v_view, dudx_view, dudy_view, dudt_view);
  }
  // TODO add kernels to compute boundary dudt boundary values?

  auto rhs_hu = [](auto args) {
    auto [u, e, h, out] = args;
    out(0, 0) = 0.5*(e(0, 0) + h(0, 0) + e(1, 0) + h(1, 0)) * u(0, 0);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(u, 0)-1, shape(u, 1)};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    dr::mhp::stencil_for_each(rhs_hu, u_view, e_view, h_view, hu_view);
  }
  dr::mhp::halo(hu).exchange_begin();

  auto rhs_hv = [](auto args) {
    auto [v, e, h, out] = args;
    out(0, 0) = 0.5*(e(0, 0) + h(0, 0) + e(0, -1) + h(0, -1)) * v(0, 0);
  };
  {
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{shape(v, 0), shape(v, 1)-1};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    dr::mhp::stencil_for_each(rhs_hv, v_view, e_view, h_view, hv_view);
  }

  dr::mhp::halo(dvdx).exchange_finalize();
  auto rhs_dvdt = [dt, g, f, dy_inv](auto tuple) {
    auto [e, u, v, dvdx, dvdy, out] = tuple;
    auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
    auto u_at_v = 0.25 * (u(0, 0) + u(0, -1) + u(-1, 0) + u(-1, -1));
    auto dvdx_up = u_at_v > 0 ? dvdx(-1, 0) : dvdx(0, 0);
    auto dvdy_up = v(0, 0) > 0 ? dvdy(0, 0) : dvdy(0, 1);
    out(0, 0) = dt * (-g * dedy - f * u_at_v - u_at_v * dvdx_up - v(0, 0) * dvdy_up);
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{shape(e, 0), shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dvdx_view = dr::mhp::views::submdspan(dvdx.view(), start, end);
    auto dvdy_view = dr::mhp::views::submdspan(dvdy.view(), start, end);
    auto dvdt_view = dr::mhp::views::submdspan(dvdt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dvdt, e_view, u_view, v_view, dvdx_view, dvdy_view, dvdt_view);
  }

  dr::mhp::halo(hu).exchange_finalize();
  auto rhs_div = [dt, dx_inv, dy_inv](auto args) {
    auto [hu, hv, out] = args;
    auto dhudx = (hu(0, 0) - hu(-1, 0)) * dx_inv;
    auto dhvdy = (hv(0, 1) - hv(0, 0)) * dy_inv;
    out(0, 0) = -dt * (dhudx + dhvdy);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(u, 0), shape(u, 1)};
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto dedt_view = dr::mhp::views::submdspan(dedt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_div, hu_view, hv_view, dedt_view);
  }
};

void rhs_vinv(Array &u, Array &v, Array &e, Array &hu, Array &hv,
         Array &dudy, Array &dvdx,
         Array &dudx, Array &dvdy,
         Array &H_at_f, Array &q, Array &ke,
         Array &dudt, Array &dvdt, Array &dedt,
         Array &h, double g, double f, double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate right hand side of the equations, vector invariant form
   */

  dr::mhp::halo(e).exchange_finalize();
  // H_at_f: average over 4 adjacent T points
  {  // interior part
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.25 * (e(1, 0) + e(1, -1) + e(0, 0) + e(0, -1));
      auto h_f = 0.25 * (h(1, 0) + h(1, -1) + h(0, 0) + h(0, -1));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{shape(e, 0) - 1, shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // top row
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(1, 0) + e(1, -1));
      auto h_f = 0.5 * (h(1, 0) + h(1, -1));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{1, shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // bottom row
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(0, 0) + e(0, -1));
      auto h_f = 0.5 * (h(0, 0) + h(0, -1));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{shape(e, 0)-1, 1};
    std::array<std::size_t, 2> end{shape(e, 0), shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // left column
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(1, 0) + e(0, 0));
      auto h_f = 0.5 * (h(1, 0) + h(0, 0));
      out(0, 0) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(e, 0)-1, 1};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // right column
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      auto e_f = 0.5 * (e(1, 0) + e(0, 0));
      auto h_f = 0.5 * (h(1, 0) + h(0, 0));
      out(0, 1) = e_f + h_f;
    };
    std::array<std::size_t, 2> start{1, shape(e, 1)-1};
    std::array<std::size_t, 2> end{shape(e, 0)-1, shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // corner (0, 0)
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 0) = e(1, 0) + h(1, 0);
    };
    std::array<std::size_t, 2> start{0, 0};
    std::array<std::size_t, 2> end{1, 1};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // corner (0, end)
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 1) = e(1, 0) + h(1, 0);
    };
    std::array<std::size_t, 2> start{0, shape(e, 1)-1};
    std::array<std::size_t, 2> end{1, shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // corner (end, 0)
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 0) = e(0, 0) + h(0, 0);
    };
    std::array<std::size_t, 2> start{shape(e, 0)-1, 0};
    std::array<std::size_t, 2> end{shape(e, 0), 1};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }
  {  // corner (end, end)
    auto rhs_Hf = [](auto tuple) {
      auto [e, h, out] = tuple;
      out(0, 1) = e(0, 0) + h(0, 0);
    };
    std::array<std::size_t, 2> start{shape(e, 0)-1, shape(e, 1)-1};
    std::array<std::size_t, 2> end{shape(e, 0), shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    dr::mhp::stencil_for_each(rhs_Hf, e_view, h_view, Hf_view);
  }

  auto rhs_dudy = [dy_inv](auto args) {
    auto [u, out] = args;
    out(0, 0) = (u(0, 0) - u(0, -1)) * dy_inv;
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{shape(dudy, 0), shape(dudy, 1)-1};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto dudy_view = dr::mhp::views::submdspan(dudy.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dudy, u_view, dudy_view);
  }

  dr::mhp::halo(v).exchange_finalize();
  auto rhs_dvdx = [dx_inv](auto args) {
    auto [v, out] = args;
    out(0, 0) = (v(1, 0) - v(0, 0)) * dx_inv;
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(dvdx, 0)-1, shape(dvdx, 1)};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dvdx_view = dr::mhp::views::submdspan(dvdx.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dvdx, v_view, dvdx_view);
  }
  dr::mhp::halo(dvdx).exchange_begin();

  // vorticity
  {
    auto kernel = [f](auto tuple) {
      auto [dudy, dvdx, H_at_f, out] = tuple;
      out(0, 0) = (f - dudy(0, 0) + dvdx(0, 0))/ H_at_f(0, 0);
    };
    std::array<std::size_t, 2> start{0, 0};
    std::array<std::size_t, 2> end{shape(q, 0), shape(q, 1)};
    // TODO full arrays -> no views needed?
    auto dudy_view = dr::mhp::views::submdspan(dudy.view(), start, end);
    auto dvdx_view = dr::mhp::views::submdspan(dvdx.view(), start, end);
    auto Hf_view = dr::mhp::views::submdspan(H_at_f.view(), start, end);
    auto q_view = dr::mhp::views::submdspan(q.view(), start, end);
    dr::mhp::stencil_for_each(kernel, dudy_view, dvdx_view, Hf_view, q_view);
  }

  // kinetic energy
  dr::mhp::halo(u).exchange_finalize();
  {
    auto kernel = [](auto tuple) {
      auto [u, v, out] = tuple;
      auto u2_at_t = 0.5 *(u(-1, 0)*u(-1, 0) + u(0, 0)*u(0, 0));
      auto v2_at_t = 0.5 *(v(0, 0)*v(0, 0) + v(0, 1)*v(0, 1));
      out(0, 0) = 0.5 * (u2_at_t + v2_at_t);
    };
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(ke, 0), shape(ke, 1)};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto ke_view = dr::mhp::views::submdspan(ke.view(), start, end);
    dr::mhp::stencil_for_each(kernel, u_view, v_view, ke_view);
  }

  auto rhs_dvdy = [dy_inv](auto args) {
    auto [v, out] = args;
    out(0, 0) = (v(0, 0) - v(0, -1)) * dy_inv;
  };
  {
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{shape(dvdy, 0), shape(dvdy, 1)};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto dvdy_view = dr::mhp::views::submdspan(dvdy.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dvdy, v_view, dvdy_view);
  }

  auto rhs_dudx = [dx_inv](auto args) {
    auto [u, out] = args;
    out(0, 0) = (u(0, 0) - u(-1, 0)) * dx_inv;
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(dudx, 0), shape(dudx, 1)};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto dudx_view = dr::mhp::views::submdspan(dudx.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dudx, u_view, dudx_view);
  }
  dr::mhp::halo(dudx).exchange_begin();

  dr::mhp::halo(dudx).exchange_finalize();
  auto rhs_dudt = [dt, g, dx_inv](auto tuple) {
    auto [e, out] = tuple;
    auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
    // auto v_at_u = 0.25 * (v(0, 0) + v(1, 0) + v(0, 1) + v(1, 1));
    // auto dudx_up = u(0, 0) > 0 ? dudx(0, 0) : dudx(1, 0);
    // auto dudy_up = v_at_u > 0 ? dudy(0, 0) : dudy(0, 1);
    // out(0, 0) = dt * (-g * dedx + f * v_at_u - u(0, 0) * dudx_up - v_at_u * dudy_up);
    out(0, 0) = dt * (-g * dedx);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(e, 0) - 1, shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    // auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    // auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    // auto dudx_view = dr::mhp::views::submdspan(dudx.view(), start, end);
    // auto dudy_view = dr::mhp::views::submdspan(dudy.view(), start, end);
    auto dudt_view = dr::mhp::views::submdspan(dudt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dudt, e_view, dudt_view);
  }
  // TODO add kernels to compute boundary dudt boundary values?

  auto rhs_hu = [](auto args) {
    auto [u, e, h, out] = args;
    out(0, 0) = 0.5*(e(0, 0) + h(0, 0) + e(1, 0) + h(1, 0)) * u(0, 0);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(u, 0)-1, shape(u, 1)};
    auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    dr::mhp::stencil_for_each(rhs_hu, u_view, e_view, h_view, hu_view);
  }
  dr::mhp::halo(hu).exchange_begin();

  auto rhs_hv = [](auto args) {
    auto [v, e, h, out] = args;
    out(0, 0) = 0.5*(e(0, 0) + h(0, 0) + e(0, -1) + h(0, -1)) * v(0, 0);
  };
  {
    std::array<std::size_t, 2> start{1, 1};
    std::array<std::size_t, 2> end{shape(v, 0), shape(v, 1)-1};
    auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    auto h_view = dr::mhp::views::submdspan(h.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    dr::mhp::stencil_for_each(rhs_hv, v_view, e_view, h_view, hv_view);
  }

  dr::mhp::halo(dvdx).exchange_finalize();
  auto rhs_dvdt = [dt, g, dy_inv](auto tuple) {
    auto [e, out] = tuple;
    auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
    // auto u_at_v = 0.25 * (u(0, 0) + u(0, -1) + u(-1, 0) + u(-1, -1));
    // auto dvdx_up = u_at_v > 0 ? dvdx(-1, 0) : dvdx(0, 0);
    // auto dvdy_up = v(0, 0) > 0 ? dvdy(0, 0) : dvdy(0, 1);
    // out(0, 0) = dt * (-g * dedy - f * u_at_v - u_at_v * dvdx_up - v(0, 0) * dvdy_up);
    out(0, 0) = dt * (-g * dedy);
  };
  {
    std::array<std::size_t, 2> start{0, 1};
    std::array<std::size_t, 2> end{shape(e, 0), shape(e, 1)};
    auto e_view = dr::mhp::views::submdspan(e.view(), start, end);
    // auto u_view = dr::mhp::views::submdspan(u.view(), start, end);
    // auto v_view = dr::mhp::views::submdspan(v.view(), start, end);
    // auto dvdx_view = dr::mhp::views::submdspan(dvdx.view(), start, end);
    // auto dvdy_view = dr::mhp::views::submdspan(dvdy.view(), start, end);
    auto dvdt_view = dr::mhp::views::submdspan(dvdt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_dvdt, e_view, dvdt_view);
  }

  dr::mhp::halo(hu).exchange_finalize();
  auto rhs_div = [dt, dx_inv, dy_inv](auto args) {
    auto [hu, hv, out] = args;
    auto dhudx = (hu(0, 0) - hu(-1, 0)) * dx_inv;
    auto dhvdy = (hv(0, 1) - hv(0, 0)) * dy_inv;
    out(0, 0) = -dt * (dhudx + dhvdy);
  };
  {
    std::array<std::size_t, 2> start{1, 0};
    std::array<std::size_t, 2> end{shape(u, 0), shape(u, 1)};
    auto hu_view = dr::mhp::views::submdspan(hu.view(), start, end);
    auto hv_view = dr::mhp::views::submdspan(hv.view(), start, end);
    auto dedt_view = dr::mhp::views::submdspan(dedt.view(), start, end);
    dr::mhp::stencil_for_each(rhs_div, hu_view, hv_view, dedt_view);
  }
};

int run(
    int n, bool benchmark_mode,
    std::function<void()> iter_callback = []() {}) {
  // construct grid
  // number of cells in x, y direction
  std::size_t nx = n;
  std::size_t ny = n;
  const double xmin = -1, xmax = 1;
  const double ymin = -1, ymax = 1;
  const double lx = xmax - xmin;
  const double ly = ymax - ymin;
  const double dx = lx / nx;
  const double dy = ly / ny;
  const double dx_inv = 1.0 / dx;
  const double dy_inv = 1.0 / dy;
  ArakawaCGrid grid(xmin, xmax, ymin, ymax, nx, ny);

  std::size_t halo_radius = 1;
  auto dist = dr::mhp::distribution().halo(halo_radius);

  // statistics
  std::size_t nread, nwrite, nflop;
  calculate_complexity(nx, ny, nread, nwrite, nflop);

  if (comm_rank == 0) {
    std::cout << "Using backend: dr" << std::endl;
    std::cout << "Grid size: " << nx << " x " << ny << std::endl;
    std::cout << "Elevation DOFs: " << nx * ny << std::endl;
    std::cout << "Velocity  DOFs: " << (nx + 1) * ny + nx * (ny + 1)
              << std::endl;
    std::cout << "Total     DOFs: " << nx * ny + (nx + 1) * ny + nx * (ny + 1);
    std::cout << std::endl;
  }

  double t_end = 0.02;
  double t_export = 0.02;

  // state variables
  // water elevation at T points
  Array e({nx + 1, ny}, dist);
  dr::mhp::fill(e, 0.0);
  // x velocity at U points
  Array u({nx + 1, ny}, dist);
  dr::mhp::fill(u, 0.0);
  // y velocity at V points
  Array v({nx + 1, ny + 1}, dist);
  dr::mhp::fill(v, 0.0);

  // bathymetry (water depth at rest) at T points
  Array h({nx + 1, ny}, dist);
  // total depth, e + h, at T points
  Array H({nx + 1, ny}, dist);

  Array hu({nx + 1, ny}, dist);
  dr::mhp::fill(hu, 0.0);
  Array hv({nx + 1, ny + 1}, dist);
  dr::mhp::fill(hu, 0.0);

  Array dudy({nx + 1, ny + 1}, dist);
  dr::mhp::fill(dudy, 0.0);
  Array dvdx({nx + 1, ny + 1}, dist);
  dr::mhp::fill(dvdx, 0.0);
  Array dudx({nx + 1, ny}, dist);
  dr::mhp::fill(dudx, 0.0);
  Array dvdy({nx + 1, ny + 1}, dist);
  dr::mhp::fill(dvdy, 0.0);

  // total depth at F points
  Array H_at_f({nx + 1, ny + 1}, dist);
  dr::mhp::fill(H_at_f, 0.0);
  // potential vorticity
  Array q({nx + 1, ny + 1}, dist);
  dr::mhp::fill(q, 0.0);
  // kinetic energy
  Array ke({nx + 1, ny}, dist);
  dr::mhp::fill(ke, 0.0);

  // set bathymetry
  set_field(h, bathymetry, grid, grid.dx / 2, grid.dy / 2, 1);
  dr::mhp::halo(h).exchange();

  // state for RK stages
  Array e1({nx + 1, ny}, dist);
  Array u1({nx + 1, ny}, dist);
  Array v1({nx + 1, ny + 1}, dist);
  Array e2({nx + 1, ny}, dist);
  Array u2({nx + 1, ny}, dist);
  Array v2({nx + 1, ny + 1}, dist);

  // time tendencies
  // NOTE not needed if rhs kernels are fused with RK stage assignment
  Array dedt({nx + 1, ny}, dist);
  Array dudt({nx + 1, ny}, dist);
  Array dvdt({nx + 1, ny + 1}, dist);

  // compute time step
  auto max = [](double x, double y) { return std::max(x, y); };
  double h_max = dr::mhp::reduce(h, static_cast<T>(0), max);
  double c = std::sqrt(g * h_max);
  double alpha = 0.5;
  double dt = alpha * dx / c;
  dt = t_export / static_cast<int>(ceil(t_export / dt));
  std::size_t nt = static_cast<int>(ceil(t_end / dt));
  if (benchmark_mode) {
    nt = 100;
    dt = 1e-5;
    t_export = 25 * dt;
    t_end = nt * dt;
  }
  if (comm_rank == 0) {
    std::cout << "Time step: " << dt << " s" << std::endl;
    std::cout << "Total run time: " << std::fixed << std::setprecision(1);
    std::cout << t_end << " s, ";
    std::cout << nt << " time steps" << std::endl;
  }

  // set initial conditions
  set_field(e, initial_elev, grid, grid.dx / 2, grid.dy / 2, 1);
  dr::mhp::halo(e).exchange_begin();
  set_field(u, initial_u, grid, 0.0, grid.dy / 2, 0);
  dr::mhp::halo(u).exchange_begin();
  set_field(v, initial_v, grid, grid.dx / 2, 0.0, 1);
  dr::mhp::halo(v).exchange_begin();

  // printArray(h, "Bathymetry");
  // printArray(e, "Initial elev");
  // printArray(u, "Initial u");
  // printArray(v, "Initial v");

  auto add = [](auto ops) { return ops.first + ops.second; };
  auto rk_update2 = [](auto ops) {
    return 0.75 * std::get<0>(ops) +
           0.25 * (std::get<1>(ops) + std::get<2>(ops));
  };
  auto rk_update3 = [](auto ops) {
    return 1.0 / 3.0 * std::get<0>(ops) +
           2.0 / 3.0 * (std::get<1>(ops) + std::get<2>(ops));
  };

  std::size_t i_export = 0;
  double next_t_export = 0.0;
  double t = 0.0;
  double initial_v = 0.0;
  auto tic = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < nt + 1; i++) {
    t = i * dt;

    if (t >= next_t_export - 1e-8) {

      double elev_max = dr::mhp::reduce(e, static_cast<T>(0), max);
      double u_max = dr::mhp::reduce(u, static_cast<T>(0), max);

      // compute total depth and volume
      dr::mhp::transform(dr::mhp::views::zip(e, h), H.begin(), add);
      double total_v =
          (dr::mhp::reduce(H, static_cast<T>(0), std::plus{})) * dx * dy;
      if (i == 0) {
        initial_v = total_v;
      }
      double diff_v = total_v - initial_v;

      if (comm_rank == 0) {
        printf("%2lu %4lu %.3f ", i_export, i, t);
        printf("elev=%7.8f ", elev_max);
        printf("u=%7.8f ", u_max);
        printf("dV=% 6.3e ", diff_v);
        printf("\n");
      }
      if (elev_max > 1e3) {
        if (comm_rank == 0) {
          std::cout << "Invalid elevation value: " << elev_max << std::endl;
        }
        return 1;
      }
      i_export += 1;
      next_t_export = i_export * t_export;
    }

    // step
    iter_callback();
    // RK stage 1: u1 = u + dt*rhs(u)
    rhs_vinv(u, v, e, hu, hv, dudy, dvdx, dudx, dvdy, H_at_f, q, ke, dudt, dvdt, dedt, h, g, f, dx_inv, dy_inv, dt);
    dr::mhp::transform(dr::mhp::views::zip(u, dudt), u1.begin(), add);
    dr::mhp::halo(u1).exchange_begin();
    dr::mhp::transform(dr::mhp::views::zip(v, dvdt), v1.begin(), add);
    dr::mhp::halo(v1).exchange_begin();
    dr::mhp::transform(dr::mhp::views::zip(e, dedt), e1.begin(), add);
    dr::mhp::halo(e1).exchange_begin();

    // RK stage 2: u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
    rhs_vinv(u1, v1, e1, hu, hv, dudy, dvdx, dudx, dvdy, H_at_f, q, ke, dudt, dvdt, dedt, h, g, f, dx_inv, dy_inv, dt);
    dr::mhp::transform(dr::mhp::views::zip(u, u1, dudt), u2.begin(),
                        rk_update2);
    dr::mhp::halo(u2).exchange_begin();
    dr::mhp::transform(dr::mhp::views::zip(v, v1, dvdt), v2.begin(),
                        rk_update2);
    dr::mhp::halo(v2).exchange_begin();
    dr::mhp::transform(dr::mhp::views::zip(e, e1, dedt), e2.begin(),
                        rk_update2);
    dr::mhp::halo(e2).exchange_begin();

    // RK stage 3: u3 = 1/3*u + 2/3*(u2 + dt*rhs(u2))
    rhs_vinv(u2, v2, e2, hu, hv, dudy, dvdx, dudx, dvdy, H_at_f, q, ke, dudt, dvdt, dedt, h, g, f, dx_inv, dy_inv, dt);
    dr::mhp::transform(dr::mhp::views::zip(u, u2, dudt), u.begin(),
                        rk_update3);
    dr::mhp::halo(u).exchange_begin();
    dr::mhp::transform(dr::mhp::views::zip(v, v2, dvdt), v.begin(),
                        rk_update3);
    dr::mhp::halo(v).exchange_begin();
    dr::mhp::transform(dr::mhp::views::zip(e, e2, dedt), e.begin(),
                        rk_update3);
    dr::mhp::halo(e).exchange_begin();
  }
  dr::mhp::halo(u).exchange_finalize();
  dr::mhp::halo(v).exchange_finalize();
  dr::mhp::halo(e).exchange_finalize();
  auto toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  if (comm_rank == 0) {
    double t_cpu = duration.count();
    double t_step = t_cpu / nt;
    double read_bw = double(nread) / t_step / (1024 * 1024 * 1024);
    double write_bw = double(nwrite) / t_step / (1024 * 1024 * 1024);
    double flop_rate = double(nflop) / t_step / (1000 * 1000 * 1000);
    std::cout << "Duration: " << std::setprecision(3) << t_cpu;
    std::cout << " s" << std::endl;
    std::cout << "Time per step: " << std::setprecision(2) << t_step * 1000;
    std::cout << " ms" << std::endl;
    std::cout << "Reads : " << std::setprecision(3) << read_bw;
    std::cout << " GB/s" << std::endl;
    std::cout << "Writes: " << std::setprecision(3) << write_bw;
    std::cout << " GB/s" << std::endl;
    std::cout << "FLOP/s: " << std::setprecision(3) << flop_rate;
    std::cout << " GFLOP/s" << std::endl;
  }

  // printArray(e, "Final elev");
  // printArray(H_at_f, "Final H_at_f");
  // printArray(q, "Final q");
  printArray(ke, "Final ke");

  // Compute error against exact solution
  Array e_exact({nx + 1, ny}, dist);
  dr::mhp::fill(e_exact, 0.0);
  Array error({nx + 1, ny}, dist);
  // initial condition for elevation
  for (auto segment : dr::ranges::segments(e_exact)) {
    if (dr::ranges::rank(segment) == std::size_t(comm_rank)) {
      auto origin = segment.origin();
      auto e = segment.mdspan();

      for (std::size_t i = 0; i < e.extent(0); i++) {
        std::size_t global_i = i + origin[0];
        if (global_i > 0) {
          for (std::size_t j = 0; j < e.extent(1); j++) {
            std::size_t global_j = j + origin[1];
            T x = xmin + dx / 2 + (global_i - 1) * dx;
            T y = ymin + dy / 2 + global_j * dy;
            e(i, j) = exact_elev(x, y, t, lx, ly);
          }
        }
      }
    }
  }
  dr::mhp::halo(e_exact).exchange();
  auto error_kernel = [](auto ops) {
    auto err = ops.first - ops.second;
    return err * err;
  };
  dr::mhp::transform(dr::mhp::views::zip(e, e_exact), error.begin(),
                     error_kernel);
  double err_L2 = dr::mhp::reduce(error, static_cast<T>(0), std::plus{}) * dx *
                  dy / lx / ly;
  err_L2 = std::sqrt(err_L2);
  if (comm_rank == 0) {
    std::cout << "L2 error: " << std::setw(7) << std::scientific;
    std::cout << std::setprecision(5) << err_L2 << std::endl;
  }

  if (benchmark_mode) {
    return 0;
  }
  if (nx < 128 || ny < 128) {
    if (comm_rank == 0) {
      std::cout << "Skipping correctness test due to small problem size."
                << std::endl;
    }
  } else if (nx == 128 && ny == 128) {
    double expected_L2 = 0.007224068445111;
    double rel_tolerance = 1e-6;
    double rel_err = err_L2 / expected_L2 - 1.0;
    if (!(fabs(rel_err) < rel_tolerance)) {
      if (comm_rank == 0) {
        std::cout << "ERROR: L2 error deviates from reference value: "
                  << expected_L2 << ", relative error: " << rel_err
                  << std::endl;
      }
      return 1;
    }
  } else {
    double tolerance = 1e-2;
    if (!(err_L2 < tolerance)) {
      if (comm_rank == 0) {
        std::cout << "ERROR: L2 error exceeds tolerance: " << err_L2 << " > "
                  << tolerance << std::endl;
      }
      return 1;
    }
  }
  if (comm_rank == 0) {
    std::cout << "SUCCESS" << std::endl;
  }

  return 0;
}

} // namespace ShallowWater

#ifdef STANDALONE_BENCHMARK

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  cxxopts::Options options_spec(argv[0], "wave equation");
  // clang-format off
  options_spec.add_options()
    ("n", "Grid size", cxxopts::value<std::size_t>()->default_value("128"))
    ("t,benchmark-mode", "Run a fixed number of time steps.", cxxopts::value<bool>()->default_value("false"))
    ("sycl", "Execute on SYCL device")
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  if (options.count("sycl")) {
#ifdef SYCL_LANGUAGE_VERSION
    sycl::queue q = dr::mhp::select_queue();
    std::cout << "Run on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    dr::mhp::init(q);
#else
    std::cout << "Sycl support requires icpx\n";
    exit(1);
#endif
  } else {
    if (comm_rank == 0) {
      std::cout << "Run on: CPU\n";
    }
    dr::mhp::init();
  }

  std::size_t n = options["n"].as<std::size_t>();
  bool benchmark_mode = options["t"].as<bool>();

  auto error = ShallowWater::run(n, benchmark_mode);
  dr::mhp::finalize();
  MPI_Finalize();
  return error;
}

#else

static void ShallowWater_DR(benchmark::State &state) {

  int n = 4000;
  std::size_t nread, nwrite, nflop;
  ShallowWater::calculate_complexity(n, n, nread, nwrite, nflop);
  Stats stats(state, nread, nwrite, nflop);

  auto iter_callback = [&stats]() { stats.rep(); };
  for (auto _ : state) {
    ShallowWater::run(n, true, iter_callback);
  }
}

DR_BENCHMARK(ShallowWater_DR);

#endif
