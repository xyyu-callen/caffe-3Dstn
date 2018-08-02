/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/tridim_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TridimMaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_d, const int stride, const int stride_d,
    const int pad_, const int pad_d_, Dtype* top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pl = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;
    int hstart = ph * stride - pad_;
    int hend = min(hstart + kernel_size, height);
    int wstart = pw * stride - pad_;
    int wend = min(wstart + kernel_size, width);
    int lstart = pl * stride_d - pad_d_;
    int lend = min(lstart + kernel_d, length);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    lstart = max(lstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
      bottom_data + (n * channels + c) * length * height * width;
    for (int l = lstart; l < lend; ++l) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (bottom_slice[l * height * width + h * width + w] > maxval) {
            maxidx = l * height * width + h * width + w;
            maxval = bottom_slice[maxidx];
          }
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] =maxval;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void TridimAvePoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_d, const int stride, const int stride_d, const int pad,
    const int pad_d_, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pl = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;
    int hstart = ph * stride - pad;
    int wstart = pw * stride - pad;
    int lstart = pl * stride_d - pad_d_;
    int hend = min(hstart + kernel_size, height + pad);
    int wend = min(wstart + kernel_size, width + pad);
    int lend = min(lstart + kernel_d, length + pad_d_);
    int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    lstart = max(lstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    lend = min(lend, length);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
      bottom_data + (n * channels + c) * length * height * width;
    for (int l = lstart; l < lend; ++l) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[(l * height + h) * width + w];
        }
      }
    }
    top_data[index] = aveval / pool_size;

  }
}

template <typename Dtype>
__global__ void TridimSumPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_d, const int stride, const int stride_d, const int pad,
    const int pad_d_, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pl = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;
    int hstart = ph * stride - pad;
    int wstart = pw * stride - pad;
    int lstart = pl * stride_d - pad_d_;
    int hend = min(hstart + kernel_size, height + pad);
    int wend = min(wstart + kernel_size, width + pad);
    int lend = min(lstart + kernel_d, length + pad_d_);
    int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    lstart = max(lstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    lend = min(lend, length);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
      bottom_data + (n * channels + c) * length * height * width;
    for (int l = lstart; l < lend; ++l) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[(l * height + h) * width + w];
        }
      }
    }
    // top_data[index] = aveval / pool_size;
    top_data[index] = aveval;

  }
}



template <typename Dtype>
void TridimPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.tridim_pooling_param().pool()) {
  case TridimPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    TridimMaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_d_,
        stride_, stride_d_, pad_, pad_d_, top_data, mask, top_mask);
    break;
  case TridimPoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    TridimAvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_d_,
        stride_, stride_d_, pad_, pad_d_, top_data);
    break;
  case TridimPoolingParameter_PoolMethod_SUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    TridimSumPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_d_,
        stride_, stride_d_, pad_, pad_d_, top_data);
    break;
  case TridimPoolingParameter_PoolMethod_STOCHASTIC:
    // NOT IMPLEMENTED YET
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
__global__ void TridimMaxPoolBackward(const int nthreads, const int* const mask,
    const Dtype* const top_mask, const Dtype* const top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_d, const int stride, const int stride_d,
    const int pad, const int pad_d, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int l = (index / width / height) % length;
    int c = (index / width / height / length) % channels;
    int n = index / width / height / length / channels;
    
    const int phstart = (h + pad < kernel_size) ? 0 : (h + pad - kernel_size) / stride + 1;
    const int phend = min((h + pad) / stride + 1, pooled_height);
    const int pwstart = (w + pad < kernel_size) ? 0 : (w + pad - kernel_size) / stride + 1;
    const int pwend = min((w + pad) / stride + 1, pooled_width);
    const int plstart = (l + pad_d < kernel_d) ? 0 : (l + pad_d - kernel_d) / stride_d + 1;
    const int plend = min((l + pad_d) / stride_d + 1, pooled_length);
    
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_length * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int pl =plstart; pl < plend; ++pl) {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            if (mask_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw]
              == l * height * width + h * width + w) {
              gradient += top_diff_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw];
            }
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int pl =plstart; pl < plend; ++pl) {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            if (top_mask_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw]
              == l * height * width + h * width + w) {
              gradient += top_diff_slice[pl * pooled_height * pooled_width + ph * pooled_width + pw];
            }
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void TridimAvePoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_d, const int stride, const int stride_d, const int pad,
    const int pad_d, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int l = (index / width / height) % length;
    int c = (index / width / height / length) % channels;
    int n = index / width / height / length / channels;
    int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    int plstart = (l < kernel_d) ? 0 : (l - kernel_d) / stride_d + 1;
    int plend = min(l / stride_d + 1, pooled_length);
    
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pooled_length * pooled_height * pooled_width;
    for (int pl = plstart; pl < plend; ++pl) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int hstart = ph * stride - pad;
          int wstart = pw * stride - pad;
          int lstart = pl * stride_d - pad_d;
          int hend = min(hstart + kernel_size, height + pad);
          int wend = min(wstart + kernel_size, width + pad);
          int lend = min(lstart + kernel_d, length + pad_d);
          int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
          gradient += top_diff[(pl * pooled_height + ph) * pooled_width + pw] / pool_size;
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void TridimSumPoolBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int length, const int height,
    const int width, const int pooled_length, const int pooled_height, const int pooled_width,
    const int kernel_size, const int kernel_d, const int stride, const int stride_d, const int pad,
    const int pad_d, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int l = (index / width / height) % length;
    int c = (index / width / height / length) % channels;
    int n = index / width / height / length / channels;
    int phstart = (h < kernel_size) ? 0 : (h - kernel_size) / stride + 1;
    int phend = min(h / stride + 1, pooled_height);
    int pwstart = (w < kernel_size) ? 0 : (w - kernel_size) / stride + 1;
    int pwend = min(w / stride + 1, pooled_width);
    int plstart = (l < kernel_d) ? 0 : (l - kernel_d) / stride_d + 1;
    int plend = min(l / stride_d + 1, pooled_length);
    
    Dtype gradient = 0;
    top_diff += (n * channels + c) * pooled_length * pooled_height * pooled_width;
    for (int pl = plstart; pl < plend; ++pl) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int hstart = ph * stride - pad;
          int wstart = pw * stride - pad;
          int lstart = pl * stride_d - pad_d;
          int hend = min(hstart + kernel_size, height + pad);
          int wend = min(wstart + kernel_size, width + pad);
          int lend = min(lstart + kernel_d, length + pad_d);
          int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
          // gradient += top_diff[(pl * pooled_height + ph) * pooled_width + pw] / pool_size;
          gradient += top_diff[(pl * pooled_height + ph) * pooled_width + pw];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void TridimPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.tridim_pooling_param().pool()) {
  case TridimPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    TridimMaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, mask, top_mask, top_diff,
        top[0]->shape(0), channels_, length_, height_, width_, pooled_length_, pooled_height_,
        pooled_width_, kernel_size_, kernel_d_, stride_, stride_d_,pad_, pad_d_, bottom_diff);
    break;
  case TridimPoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    TridimAvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->shape(0), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_d_, 
        stride_, stride_d_, pad_, pad_d_, bottom_diff);
    break;
  case TridimPoolingParameter_PoolMethod_SUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    TridimSumPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->shape(0), channels_, length_,
        height_, width_, pooled_length_, pooled_height_, pooled_width_, kernel_size_, kernel_d_, 
        stride_, stride_d_, pad_, pad_d_, bottom_diff);
    break;
  case TridimPoolingParameter_PoolMethod_STOCHASTIC:
    // NOT IMPLEMENTED YET
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(TridimPoolingLayer);


}  // namespace caffe
