// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <CL/sycl.hpp>
#include <dr/shp/device_ptr.hpp>
#include <dr/shp/device_vector.hpp>
#include <vector>

#include <dr/details/segments_tools.hpp>
#include <dr/shp/allocators.hpp>
#include <dr/shp/vector.hpp>

namespace shp {

template <typename T, typename L> class distributed_vector_accessor {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;

  using segment_type = L;
  using const_segment_type = std::add_const_t<L>;
  using nonconst_segment_type = std::remove_const_t<L>;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  // using pointer = typename segment_type::pointer;
  using reference = rng::range_reference_t<segment_type>;

  using iterator_category = std::random_access_iterator_tag;

  using iterator_accessor = distributed_vector_accessor;
  using const_iterator_accessor = iterator_accessor;
  using nonconst_iterator_accessor = iterator_accessor;

  constexpr distributed_vector_accessor() noexcept = default;
  constexpr ~distributed_vector_accessor() noexcept = default;
  constexpr distributed_vector_accessor(
      const distributed_vector_accessor &) noexcept = default;
  constexpr distributed_vector_accessor &
  operator=(const distributed_vector_accessor &) noexcept = default;

  constexpr distributed_vector_accessor(std::span<segment_type> segments,
                                        size_type segment_id, size_type idx,
                                        size_type segment_size) noexcept
      : segments_(segments), segment_id_(segment_id), idx_(idx),
        segment_size_(segment_size) {}

  constexpr distributed_vector_accessor &
  operator+=(difference_type offset) noexcept {
    if (offset > 0) {
      idx_ += offset;
      if (idx_ >= segment_size_) {
        segment_id_ += idx_ / segment_size_;
        idx_ = idx_ % segment_size_;
      }
    }

    if (offset < 0) {
      size_type new_global_idx = get_global_idx() + offset;
      segment_id_ = new_global_idx / segment_size_;
      idx_ = new_global_idx % segment_size_;
    }
    return *this;
  }

  constexpr bool operator==(const iterator_accessor &other) const noexcept {
    return segment_id_ == other.segment_id_ && idx_ == other.idx_;
  }

  constexpr difference_type
  operator-(const iterator_accessor &other) const noexcept {
    return difference_type(get_global_idx()) - other.get_global_idx();
  }

  constexpr bool operator<(const iterator_accessor &other) const noexcept {
    if (segment_id_ < other.segment_id_) {
      return true;
    } else if (segment_id_ == other.segment_id_) {
      return idx_ < other.idx_;
    } else {
      return false;
    }
  }

  constexpr reference operator*() const noexcept {
    return segments_[segment_id_][idx_];
  }

  auto segments() const noexcept {
    return lib::internal::drop_segments(segments_, segment_id_, idx_);
  }

private:
  size_type get_global_idx() const noexcept {
    return segment_id_ * segment_size_ + idx_;
  }

  std::span<segment_type> segments_;
  size_type segment_id_ = 0;
  size_type idx_ = 0;
  size_type segment_size_ = 0;
};

template <typename T, typename L>
using distributed_vector_iterator =
    lib::iterator_adaptor<distributed_vector_accessor<T, L>>;

// TODO: support teams, distributions

template <typename T, typename Allocator = shp::shared_allocator<T>>
struct distributed_vector {
public:
  using segment_type = shp::device_vector<T, Allocator>;
  using const_segment_type = std::add_const_t<shp::device_vector<T, Allocator>>;

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using pointer = decltype(std::declval<segment_type>().data());
  using const_pointer =
      decltype(std::declval<std::add_const_t<segment_type>>().data());

  using reference = std::iter_reference_t<pointer>;
  using const_reference = std::iter_reference_t<const_pointer>;

  using iterator = distributed_vector_iterator<T, segment_type>;
  using const_iterator =
      distributed_vector_iterator<const T, const_segment_type>;

  distributed_vector(std::size_t count = 0) {
    assert(shp::devices().size() > 0);
    size_ = count;
    segment_size_ = (count + shp::devices().size() - 1) / shp::devices().size();
    capacity_ = segment_size_ * shp::devices().size();

    size_t rank = 0;
    for (auto &&device : shp::devices())
      segments_.emplace_back(segment_type(
          segment_size_, Allocator(shp::context(), device), rank++));
  }

  distributed_vector(std::size_t count, value_type fill_value)
      : distributed_vector(count) {
    std::vector<cl::sycl::event> events;

    for (auto &&segment : segments_)
      events.push_back(
          shp::fill_async(segment.begin(), segment.end(), fill_value));

    sycl::queue().submit([=](auto &&h) { h.depends_on(events); }).wait();
  }

  distributed_vector(std::initializer_list<value_type> __l)
      : distributed_vector(__l.size()) {
    shp::copy(std::begin(__l), std::end(__l), this->begin());
  }

  reference operator[](size_type pos) {
    size_type segment_id = pos / segment_size_;
    size_type local_id = pos % segment_size_;
    return *(segments_[segment_id].begin() + local_id);
  }

  const_reference operator[](size_type pos) const {
    size_type segment_id = pos / segment_size_;
    size_type local_id = pos % segment_size_;
    return *(segments_[segment_id].begin() + local_id);
  }

  size_type size() const noexcept { return size_; }

  auto segments() { return lib::internal::take_segments(segments_, size()); }

  auto segments() const {
    return lib::internal::take_segments(segments_, size());
  }

  iterator begin() { return iterator(segments_, 0, 0, segment_size_); }

  const_iterator begin() const {
    return const_iterator(segments_, 0, 0, segment_size_);
  }

  iterator end() {
    return iterator(segments_, size() / segment_size_, size() % segment_size_,
                    segment_size_);
  }

  const_iterator end() const {
    return const_iterator(segments_, size() / segment_size_,
                          size() % segment_size_, segment_size_);
  }

private:
  std::vector<segment_type> segments_;
  std::size_t capacity_ = 0;
  std::size_t size_ = 0;
  std::size_t segment_size_ = 0;
};

} // namespace shp
