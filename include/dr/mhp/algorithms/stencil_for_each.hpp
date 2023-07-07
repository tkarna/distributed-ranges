// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <execution>
#include <type_traits>
#include <utility>

#include <dr/concepts/concepts.hpp>
#include <dr/detail/logger.hpp>
#include <dr/detail/onedpl_direct_iterator.hpp>
#include <dr/detail/ranges_shim.hpp>
#include <dr/mhp/global.hpp>

namespace dr::mhp {

/// Collective for_each on distributed range
template <typename... Ts>
void stencil_for_each(std::size_t radius, auto op,
                      dr::distributed_range auto &&dr1,
                      dr::distributed_range auto &&dr2) {
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();
  auto grid2 = dr2.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      auto t1 = grid1(tile_index, 0).mdspan();
      auto t2 = grid2(tile_index, 0).mdspan();

      // TODO support arbitrary ranks
      assert(t1.rank() == t2.rank() && t2.rank() == 2);

      // Do not update halo for first and last segment
      std::size_t first = 0 + radius * (tile_index == 0);
      std::size_t last =
          t1.extent(0) - radius * (tile_index == (grid1.extent(0) - 1));
      for (std::size_t i = first; i < last; i++) {
        for (std::size_t j = radius; j < t1.extent(1) - radius; j++) {
          auto t1_stencil =
              md::mdspan(std::to_address(&t1(i, j)), t1.extents());
          auto t2_stencil =
              md::mdspan(std::to_address(&t2(i, j)), t2.extents());
          op(std::tuple(t1_stencil, t2_stencil));
        }
      }
    }
  }

  barrier();
}

/// Version with user defined iteration range; 3 inputs
template <typename... Ts>
void stencil_for_each_fuse3(dr_extents<4> stencil_extents, auto op,
                            dr::distributed_range auto &&dr1,
                            dr::distributed_range auto &&dr2,
                            dr::distributed_range auto &&dr3) {
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();
  auto grid2 = dr2.grid();
  auto grid3 = dr3.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      auto t1 = grid1(tile_index, 0).mdspan();
      auto t2 = grid2(tile_index, 0).mdspan();
      auto t3 = grid3(tile_index, 0).mdspan();

      // TODO support arbitrary ranks
      assert(t1.rank() == t2.rank() && t2.rank() == t3.rank() &&
             t2.rank() == 2);

      // Do not update halo for first and last segment
      std::size_t first = 0 + stencil_extents[0] * (tile_index == 0);
      std::size_t last =
          t1.extent(0) -
          stencil_extents[1] * (tile_index == (grid1.extent(0) - 1));
      for (std::size_t i = first; i < last; i++) {
        for (std::size_t j = stencil_extents[2];
             j < t1.extent(1) - stencil_extents[3]; j++) {
          auto t1_stencil =
              md::mdspan(std::to_address(&t1(i, j)), t1.extents());
          auto t2_stencil =
              md::mdspan(std::to_address(&t2(i, j)), t2.extents());
          auto t3_stencil =
              md::mdspan(std::to_address(&t3(i, j)), t3.extents());
          op(std::tuple(t1_stencil, t2_stencil, t3_stencil));
        }
      }
    }
  }

  barrier();
}

/// Version with user defined iteration range; 4 inputs
template <typename... Ts>
void stencil_for_each_fuse4(dr_extents<4> stencil_extents, auto op,
                            dr::distributed_range auto &&dr1,
                            dr::distributed_range auto &&dr2,
                            dr::distributed_range auto &&dr3,
                            dr::distributed_range auto &&dr4) {
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();
  auto grid2 = dr2.grid();
  auto grid3 = dr3.grid();
  auto grid4 = dr4.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      auto t1 = grid1(tile_index, 0).mdspan();
      auto t2 = grid2(tile_index, 0).mdspan();
      auto t3 = grid3(tile_index, 0).mdspan();
      auto t4 = grid4(tile_index, 0).mdspan();

      // TODO support arbitrary ranks
      assert(t1.rank() == t2.rank() && t2.rank() == t3.rank() &&
             t3.rank() == t4.rank() && t2.rank() == 2);

      // Do not update halo for first and last segment
      std::size_t first = 0 + stencil_extents[0] * (tile_index == 0);
      std::size_t last =
          t1.extent(0) -
          stencil_extents[1] * (tile_index == (grid1.extent(0) - 1));
      for (std::size_t i = first; i < last; i++) {
        for (std::size_t j = stencil_extents[2];
             j < t1.extent(1) - stencil_extents[3]; j++) {
          auto t1_stencil =
              md::mdspan(std::to_address(&t1(i, j)), t1.extents());
          auto t2_stencil =
              md::mdspan(std::to_address(&t2(i, j)), t2.extents());
          auto t3_stencil =
              md::mdspan(std::to_address(&t3(i, j)), t3.extents());
          auto t4_stencil =
              md::mdspan(std::to_address(&t4(i, j)), t4.extents());
          op(std::tuple(t1_stencil, t2_stencil, t3_stencil, t4_stencil));
        }
      }
    }
  }

  barrier();
}

/// Version with user defined iteration range; 5 inputs
template <typename... Ts>
void stencil_for_each_fuse5(dr_extents<4> stencil_extents, auto op,
                            dr::distributed_range auto &&dr1,
                            dr::distributed_range auto &&dr2,
                            dr::distributed_range auto &&dr3,
                            dr::distributed_range auto &&dr4,
                            dr::distributed_range auto &&dr5) {
  if (rng::empty(dr1)) {
    return;
  }

  auto grid1 = dr1.grid();
  auto grid2 = dr2.grid();
  auto grid3 = dr3.grid();
  auto grid4 = dr4.grid();
  auto grid5 = dr5.grid();

  // TODO: Support distribution other than first dimension
  assert(grid1.extent(1) == 1);
  for (std::size_t tile_index = 0; tile_index < grid1.extent(0); tile_index++) {
    // If local
    if (tile_index == default_comm().rank()) {
      auto t1 = grid1(tile_index, 0).mdspan();
      auto t2 = grid2(tile_index, 0).mdspan();
      auto t3 = grid3(tile_index, 0).mdspan();
      auto t4 = grid4(tile_index, 0).mdspan();
      auto t5 = grid5(tile_index, 0).mdspan();

      // TODO support arbitrary ranks
      assert(t1.rank() == t2.rank() && t2.rank() == t3.rank() &&
             t3.rank() == t4.rank() && t4.rank() == t5.rank() &&
             t2.rank() == 2);

      // Do not update halo for first and last segment
      std::size_t first = 0 + stencil_extents[0] * (tile_index == 0);
      std::size_t last =
          t1.extent(0) -
          stencil_extents[1] * (tile_index == (grid1.extent(0) - 1));
      for (std::size_t i = first; i < last; i++) {
        for (std::size_t j = stencil_extents[2];
             j < t1.extent(1) - stencil_extents[3]; j++) {
          auto t1_stencil =
              md::mdspan(std::to_address(&t1(i, j)), t1.extents());
          auto t2_stencil =
              md::mdspan(std::to_address(&t2(i, j)), t2.extents());
          auto t3_stencil =
              md::mdspan(std::to_address(&t3(i, j)), t3.extents());
          auto t4_stencil =
              md::mdspan(std::to_address(&t4(i, j)), t4.extents());
          auto t5_stencil =
              md::mdspan(std::to_address(&t5(i, j)), t5.extents());
          op(std::tuple(t1_stencil, t2_stencil, t3_stencil, t4_stencil,
                        t5_stencil));
        }
      }
    }
  }

  barrier();
}

} // namespace dr::mhp
