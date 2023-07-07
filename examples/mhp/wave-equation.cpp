// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause
#include "cxxopts.hpp"
#include "dr/mhp.hpp"
#include "mpi.h"
#include <chrono>
#include <iomanip>

using T = double;

using Array = dr::mhp::distributed_mdarray<T, 2>;

void print_array(Array &a, std::string msg) {
  auto nx = a.mdspan().extent(0);
  auto ny = a.mdspan().extent(1);
  const auto default_precision{std::cout.precision()};
  std::cout << msg << " (" << nx << ", " << ny << "): [";
  std::cout << std::fixed << std::setprecision(5);
  for (std::size_t i = 0; i < nx; i++) {
    std::cout << "\n  ";
    for (std::size_t j = 0; j < ny; j++) {
      std::cout << std::setw(9) << a.mdspan()(i, j) << " ";
    }
  }
  std::cout << "\n]\n";
  std::cout << std::setprecision(default_precision);
}

MPI_Comm comm;
int comm_rank;
int comm_size;

// gravitational acceleration
constexpr double g = 9.81;
// water depth
constexpr double h = 1.0;

double exact_elev(double x, double y, double t, double lx, double ly) {
  /**
   * Exact solution for elevation field.
   *
   * Returns time-dependent elevation of a 2D standing wave in a
   * rectangular domain.
   */
  double amp = 0.5;
  double c = std::sqrt(g * h);
  std::size_t n = 1;
  double sol_x = std::cos(2 * n * M_PI * x / lx);
  std::size_t m = 1;
  double sol_y = std::cos(2 * m * M_PI * y / ly);
  double omega = c * M_PI * std::hypot(n / lx, m / ly);
  double sol_t = std::cos(2 * omega * t);
  return amp * sol_x * sol_y * sol_t;
}

double initial_elev(double x, double y, double lx, double ly) {
  return exact_elev(x, y, 0.0, lx, ly);
}

void stage1(Array &u, Array &v, Array &e, Array &u1, Array &v1, Array &e1,
            double g, double h, double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate stage 1 of the RK time stepper
   *
   * u1 = u + dt*rhs(u)
   *
   */
  // u: elevation x gradient
  auto rhs_u1 = [dt, g, dx_inv](auto tuple) {
    auto [e, u, out] = tuple;
    auto dedx = (e(1, 0) - e(0, 0)) * dx_inv;
    out(0, 0) = u(0, 0) - dt * g * dedx;
  };
  dr::mhp::stencil_for_each_fuse3({1, 1, 0, 0}, rhs_u1, e, u, u1);
  dr::mhp::halo(u1).exchange();

  // v: elevation y gradient
  auto rhs_v1 = [dt, g, dy_inv](auto tuple) {
    auto [e, v, out] = tuple;
    auto dedy = (e(0, 0) - e(0, -1)) * dy_inv;
    out(0, 0) = v(0, 0) - dt * g * dedy;
  };
  dr::mhp::stencil_for_each_fuse3({0, 0, 1, 0}, rhs_v1, e, v, v1);
  dr::mhp::halo(v1).exchange();

  // e: divergence of (u, v)
  auto rhs_e1 = [dt, h, dx_inv, dy_inv](auto tuple) {
    auto [e, u, v, out] = tuple;
    auto dudx = (u(0, 0) - u(-1, 0)) * dx_inv;
    auto dvdy = (v(0, 1) - v(0, 0)) * dy_inv;
    out(0, 0) = e(0, 0) - dt * h * (dudx + dvdy);
  };
  dr::mhp::stencil_for_each_fuse4({1, 0, 0, 0}, rhs_e1, e, u, v, e1);
  dr::mhp::halo(e1).exchange();
};

void stage2(Array &u, Array &v, Array &e, Array &u1, Array &v1, Array &e1,
            Array &u2, Array &v2, Array &e2, double g, double h, double dx_inv,
            double dy_inv, double dt) {
  /**
   * Evaluate stage 2 of the RK time stepper
   *
   * u2 = 0.75*u + 0.25*(u1 + dt*rhs(u1))
   *
   */
  // u: elevation x gradient
  auto rhs_u2 = [dt, g, dx_inv](auto tuple) {
    auto [e1, u1, u, out] = tuple;
    auto dedx = (e1(1, 0) - e1(0, 0)) * dx_inv;
    out(0, 0) = 0.75 * u(0, 0) + 0.25 * (u1(0, 0) - dt * g * dedx);
  };
  dr::mhp::stencil_for_each_fuse4({1, 1, 0, 0}, rhs_u2, e1, u1, u, u2);
  dr::mhp::halo(u2).exchange();

  // v: elevation y gradient
  auto rhs_v2 = [dt, g, dy_inv](auto tuple) {
    auto [e1, v1, v, out] = tuple;
    auto dedy = (e1(0, 0) - e1(0, -1)) * dy_inv;
    out(0, 0) = 0.75 * v(0, 0) + 0.25 * (v1(0, 0) - dt * g * dedy);
  };
  dr::mhp::stencil_for_each_fuse4({0, 0, 1, 0}, rhs_v2, e1, v1, v, v2);
  dr::mhp::halo(v2).exchange();

  // e: divergence of (u, v)
  auto rhs_e2 = [dt, h, dx_inv, dy_inv](auto tuple) {
    auto [e1, u1, v1, e, out] = tuple;
    auto dudx = (u1(0, 0) - u1(-1, 0)) * dx_inv;
    auto dvdy = (v1(0, 1) - v1(0, 0)) * dy_inv;
    out(0, 0) = 0.75 * e(0, 0) + 0.25 * (e1(0, 0) - dt * h * (dudx + dvdy));
  };
  dr::mhp::stencil_for_each_fuse5({1, 0, 0, 0}, rhs_e2, e1, u1, v1, e, e2);
  dr::mhp::halo(e2).exchange();
};

void stage3(Array &u, Array &v, Array &e, Array &u2, Array &v2, Array &e2,
            double g, double h, double dx_inv, double dy_inv, double dt) {
  /**
   * Evaluate stage 3 of the RK time stepper
   *
   * u3 = 1/3*u + 2/3*(u2 + dt*rhs(u2))
   *
   */
  // u: elevation x gradient
  auto rhs_u3 = [dt, g, dx_inv](auto tuple) {
    auto [e2, u2, out] = tuple;
    auto dedx = (e2(1, 0) - e2(0, 0)) * dx_inv;
    out(0, 0) *= 1.0 / 3;
    out(0, 0) += 2.0 / 3 * (u2(0, 0) - dt * g * dedx);
  };
  dr::mhp::stencil_for_each_fuse3({1, 1, 0, 0}, rhs_u3, e2, u2, u);
  dr::mhp::halo(u).exchange();

  // v: elevation y gradient
  auto rhs_v3 = [dt, g, dy_inv](auto tuple) {
    auto [e2, v2, out] = tuple;
    auto dedy = (e2(0, 0) - e2(0, -1)) * dy_inv;
    out(0, 0) *= 1.0 / 3;
    out(0, 0) += 2.0 / 3 * (v2(0, 0) - dt * g * dedy);
  };
  dr::mhp::stencil_for_each_fuse3({0, 0, 1, 0}, rhs_v3, e2, v2, v);
  dr::mhp::halo(v).exchange();

  // e: divergence of (u, v)
  auto rhs_e2 = [dt, h, dx_inv, dy_inv](auto tuple) {
    auto [e2, u2, v2, out] = tuple;
    auto dudx = (u2(0, 0) - u2(-1, 0)) * dx_inv;
    auto dvdy = (v2(0, 1) - v2(0, 0)) * dy_inv;
    out(0, 0) *= 1.0 / 3;
    out(0, 0) += 2.0 / 3 * (e2(0, 0) - dt * h * (dudx + dvdy));
  };
  dr::mhp::stencil_for_each_fuse4({1, 0, 0, 0}, rhs_e2, e2, u2, v2, e);
  dr::mhp::halo(e).exchange();
};

int run(int n, bool benchmark_mode) {

  // Arakava C grid
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
  std::size_t halo_radius = 1;
  auto dist = dr::mhp::distribution().halo(halo_radius);

  if (comm_rank == 0) {
    std::cout << "Using backend: dr" << std::endl;
    std::cout << "Grid size: " << nx << " x " << ny << std::endl;
    std::cout << "Elevation DOFs: " << nx * ny << std::endl;
    std::cout << "Velocity  DOFs: " << (nx + 1) * ny + nx * (ny + 1)
              << std::endl;
    std::cout << "Total     DOFs: " << nx * ny + (nx + 1) * ny + nx * (ny + 1);
    std::cout << std::endl;
  }

  // compute time step
  double t_end = 1.0;
  double t_export = 0.02;

  double c = std::sqrt(g * h);
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

  // state variables
  // water elevation at T points
  Array e({nx + 1, ny}, dist);
  // x velocity at U points
  Array u({nx + 1, ny}, dist);
  // y velocity at V points
  Array v({nx + 1, ny + 1}, dist);
  dr::mhp::fill(e, 0.0);
  dr::mhp::fill(u, 0.0);
  dr::mhp::fill(v, 0.0);

  // state for RK stages
  Array e1({nx + 1, ny}, dist);
  Array u1({nx + 1, ny}, dist);
  Array v1({nx + 1, ny + 1}, dist);
  dr::mhp::fill(e1, 0.0);
  dr::mhp::fill(u1, 0.0);
  dr::mhp::fill(v1, 0.0);
  Array e2({nx + 1, ny}, dist);
  Array u2({nx + 1, ny}, dist);
  Array v2({nx + 1, ny + 1}, dist);
  dr::mhp::fill(e2, 0.0);
  dr::mhp::fill(u2, 0.0);
  dr::mhp::fill(v2, 0.0);

  // initial condition for elevation
  for (std::size_t i = 1; i < e.mdspan().extent(0); i++) {
    for (std::size_t j = 0; j < e.mdspan().extent(1); j++) {
      T x = xmin + dx / 2 + (i - 1) * dx;
      T y = ymin + dy / 2 + j * dy;
      e.mdspan()(i, j) = initial_elev(x, y, lx, ly);
    }
  }
  dr::mhp::halo(e).exchange();

  auto max = [](double x, double y) { return std::max(x, y); };

  std::size_t i_export = 0;
  double next_t_export = 0.0;
  double t = 0.0;
  double initial_v;
  auto tic = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < nt + 1; i++) {
    t = i * dt;

    if (t >= next_t_export - 1e-8) {

      double elev_max = dr::mhp::reduce(e, static_cast<T>(0), max);
      double u_max = dr::mhp::reduce(u, static_cast<T>(0), max);

      double total_v =
          (dr::mhp::reduce(e, static_cast<T>(0), std::plus{}) + h) * dx * dy;
      if (i == 0) {
        initial_v = total_v;
      }
      double diff_v = total_v - initial_v;

      if (comm_rank == 0) {
        printf("%2lu %4lu %.3f ", i_export, i, t);
        printf("elev=%7.5f ", elev_max);
        printf("u=%7.5f ", u_max);
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
    stage1(u, v, e, u1, v1, e1, g, h, dx_inv, dy_inv, dt);
    stage2(u, v, e, u1, v1, e1, u2, v2, e2, g, h, dx_inv, dy_inv, dt);
    stage3(u, v, e, u2, v2, e2, g, h, dx_inv, dy_inv, dt);
  }
  auto toc = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration = toc - tic;
  if (comm_rank == 0) {
    std::cout << "Duration: " << std::setprecision(2) << duration.count();
    std::cout << " s" << std::endl;
  }

  // Compute error against exact solution
  Array e_exact({nx + 1, ny}, dist);
  dr::mhp::fill(e_exact, 0.0);
  Array error({nx + 1, ny}, dist);
  for (std::size_t i = 1; i < e_exact.mdspan().extent(0); i++) {
    for (std::size_t j = 0; j < e_exact.mdspan().extent(1); j++) {
      T x = xmin + dx / 2 + (i - 1) * dx;
      T y = ymin + dy / 2 + j * dy;
      e_exact.mdspan()(i, j) = exact_elev(x, y, t, lx, ly);
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
    if (fabs(rel_err) > rel_tolerance) {
      if (comm_rank == 0) {
        std::cout << "ERROR: L2 error deviates from reference value: "
                  << expected_L2 << ", relative error: " << rel_err
                  << std::endl;
      }
      return 1;
    }
  } else {
    double tolerance = 1e-2;
    if (err_L2 > tolerance) {
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

int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  dr::mhp::init();

  cxxopts::Options options_spec(argv[0], "wave equation");
  // clang-format off
  options_spec.add_options()
    ("n", "Grid size", cxxopts::value<std::size_t>()->default_value("128"))
    ("t,benchmark-mode", "Run a fixed number of time steps.", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  std::size_t n = options["n"].as<std::size_t>();
  bool benchmark_mode = options["t"].as<bool>();

  auto error = run(n, benchmark_mode);
  MPI_Finalize();
  return error;
}
