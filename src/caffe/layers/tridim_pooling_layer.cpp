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
#include "caffe/util/benchmark.hpp"

namespace caffe {

using std::max;
using std::min;

template <typename Dtype>
void TridimPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Jointly implemented in LayerSetUp
}

template <typename Dtype>
void TridimPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "TridimPoolingLayer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "TridimPoolingLayer takes a single blob as output.";
  kernel_size_ = this->layer_param_.tridim_pooling_param().kernel_size();
  kernel_d_ = this->layer_param_.tridim_pooling_param().kernel_size();
  //kernel_d_ = this->layer_param_.tridim_pooling_param().kernel_d();
  stride_ = this->layer_param_.tridim_pooling_param().stride();
  stride_d_ = this->layer_param_.tridim_pooling_param().stride();
  //stride_d_ = this->layer_param_.tridim_pooling_param().stride_d();
  pad_ = this->layer_param_.tridim_pooling_param().pad();
  pad_d_ = this->layer_param_.tridim_pooling_param().pad();

  if (pad_ != 0) {
    CHECK(this->layer_param_.tridim_pooling_param().pool()
    	== TridimPoolingParameter_PoolMethod_AVE
    	|| this->layer_param_.pooling3d_param().pool()
    	== Pooling3DParameter_PoolMethod_SUM)
        << "Padding implemented only for average pooling and sum pooling.";
  }

  channels_ = bottom[0]->shape(1);
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);

  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_ - kernel_size_) / stride_)) + 1;
  pooled_length_ = static_cast<int>(ceil(static_cast<float>(
	      length_ + 2 * pad_d_ - kernel_d_) / stride_d_)) + 1;
  /*
  pooled_length_ = static_cast<int>(ceil(static_cast<float>(
	      length_ - kernel_d_) / stride_d_)) + 1;
  */

  vector<int> top_shape(5);
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = channels_;
  top_shape[2] = pooled_length_;
  top_shape[3] = pooled_height_;
  top_shape[4] = pooled_width_;

  top[0]->Reshape(top_shape);
  //top[0]->Reshape(bottom[0]->num(), channels_, pooled_length_, pooled_height_, pooled_width_);

  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
  	max_idx_.Reshape(top_shape);
    // max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  }

  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.tridim_pooling_param().pool() ==
      TridimPoolingParameter_PoolMethod_STOCHASTIC) {
	  rand_idx_.Reshape(top_shape);
    //rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_length_, pooled_height_, pooled_width_);
  }
}

template <typename Dtype>
void TridimPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top) {

  CPUTimer timer;
  double computation;
  timer.Start();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
	  static const int index_array[] = {0, 1, 0, 0, 0};
	  vector<int> offset_indices(index_array, index_array + 5);

	  // Different pooling methods. We explicitly do the switch outside the for
	  // loop to save time, although this results in more codes.
	  int top_count = top[0]->count();
	  const bool use_top_mask = top.size() > 1;
	  int* mask = NULL;
	  Dtype* top_mask = NULL;
	  switch (this->layer_param_.tridim_pooling_param().pool()) {
	  case TridimPoolingParameter_PoolMethod_MAX:
	    // Initialize
		if (use_top_mask) {
			top_mask = top[1]->mutable_cpu_data();
			caffe_set(top_count, Dtype(-1), top_mask);
		} else {
			mask = max_idx_.mutable_cpu_data();
			caffe_set(top_count, -1, mask);
		}
	    for (int i = 0; i < top_count; ++i) {
	      top_data[i] = Dtype(-FLT_MAX);
	    }
	    // The main loop
	    for (int n = 0; n < bottom[0]->shape(0); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_ - pad_;
	              int wstart = pw * stride_ - pad_;
	              int lstart = pl * stride_d_ - pad_d_;
	              int hend = min(hstart + kernel_size_, height_);
	              int wend = min(wstart + kernel_size_, width_);
	              int lend = min(lstart + kernel_d_, length_);
	              hstart = max(hstart, 0);
	              wstart = max(wstart, 0);
	              lstart = max(lstart, 0);
	              const int pool_index = pl * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw;
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                  	const int index = l * height_ * width_ + h * width_ + w;
	                  	if (bottom_data[index] > top_data[pool_index]) {
		                  top_data[pool_index] = bottom_data[index];
		                  if (use_top_mask) {
		                    top_mask[pool_index] = static_cast<Dtype>(index);
		                  } else {
		                    mask[pool_index] = index;
		                  }
		                }
	                  }
	                }
	              }
	            }
	          }
	    	}
	        // compute offset
	        bottom_data += bottom[0]->offset(offset_indices);
	        top_data += top[0]->offset(offset_indices);
	        if (use_top_mask) {
	        	top_mask += top[0]->offset(offset_indices);
	        } else {
	        	mask += top[0]->offset(offset_indices);
	        }
	      }
	    }
	    break;
	  case TridimPoolingParameter_PoolMethod_AVE:
	    for (int i = 0; i < top_count; ++i) {
	      top_data[i] = 0;
	    }
	    // The main loop
	    for (int n = 0; n < bottom[0]->shape(0); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_ - pad_;
	              int wstart = pw * stride_ - pad_;
	              int lstart = pl * stride_d_ - pad_d_;
	              int hend = min(hstart + kernel_size_, height_ + pad_);
	              int wend = min(wstart + kernel_size_, width_ + pad_);
	              int lend = min(lstart + kernel_d_, length_ + pad_d_);
	              int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
	              hstart = max(hstart, 0);
	              wstart = max(wstart, 0);
	              lstart = max(lstart, 0);

	              hend = min(hend, height_);
	              wend = min(wend, width_);
	              lend = min(lend, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] +=
	                        bottom_data[(l * height_ + h) * width_ + w];
	                  }
	                }
	              }
	              top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] /= pool_size;
	            }
	          }
	    	}
	        // compute offset
	        bottom_data += bottom[0]->offset(offset_indices);
	        top_data += top[0]->offset(offset_indices);
	      }
	    }
	    break;
	  case TridimPoolingParameter_PoolMethod_SUM:
	    for (int i = 0; i < top_count; ++i) {
	      top_data[i] = 0;
	    }
	    // The main loop
	    for (int n = 0; n < bottom[0]->shape(0); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_ - pad_;
	              int wstart = pw * stride_ - pad_;
	              int lstart = pl * stride_d_ - pad_d_;
	              int hend = min(hstart + kernel_size_, height_ + pad_);
	              int wend = min(wstart + kernel_size_, width_ + pad_);
	              int lend = min(lstart + kernel_d_, length_ + pad_d_);
	              int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
	              hstart = max(hstart, 0);
	              wstart = max(wstart, 0);
	              lstart = max(lstart, 0);

	              hend = min(hend, height_);
	              wend = min(wend, width_);
	              lend = min(lend, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] +=
	                        bottom_data[(l * height_ + h) * width_ + w];
	                  }
	                }
	              }
	              // top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] /= pool_size;
	            }
	          }
	    	}
	        // compute offset
	        bottom_data += bottom[0]->offset(offset_indices);
	        top_data += top[0]->offset(offset_indices);
	      }
	    }
	    break;
	  case TridimPoolingParameter_PoolMethod_STOCHASTIC:
	    NOT_IMPLEMENTED;
	    break;
	  default:
	    LOG(FATAL) << "Unknown pooling method.";
	  }
  computation = timer.MilliSeconds();
  std::ofstream out("profile_inference.log", std::ofstream::out | std::ofstream::app);
  out << this->layer_param_.name() << " CPU " << computation << "\n";
  out.close();
}

template <typename Dtype>
void TridimPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	  if (!propagate_down[0]) {
	    return;
	  }
	  const Dtype* top_diff = top[0]->cpu_diff();
	  // const Dtype* top_data = top[0]->cpu_data();
	  // const Dtype* bottom_data = bottom[0]->cpu_data();
	  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	  static const int index_array[] = {0, 1, 0, 0, 0};
	  vector<int> offset_indices(index_array, index_array + 5);

	  // Different pooling methods. We explicitly do the switch outside the for
	  // loop to save time, although this results in more codes.
	  memset(bottom_diff, 0, bottom[0]->count() * sizeof(Dtype));
	  const bool use_top_mask = top.size() > 1;
	  const int* mask = NULL;
	  const Dtype* top_mask = NULL;
	  switch (this->layer_param_.tridim_pooling_param().pool()) {

	  case TridimPoolingParameter_PoolMethod_MAX:
	    // The main loop
	    for (int n = 0; n < top[0]->shape(0); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              const int index =  pl * pooled_height_ * pooled_width_ + ph * pooled_width_ + pw;
	              const int bottom_index = use_top_mask ? top_mask[index] : mask[index];
	              bottom_diff[bottom_index] += top_diff[index];
	            }
	          }
	    	}
	        // offset
	        bottom_diff += bottom[0]->offset(offset_indices);
	        top_diff += top[0]->offset(offset_indices);
	        if (use_top_mask) {
	        	top_mask += top[0]->offset(offset_indices);
	        } else {
	        	mask += top[0]->offset(offset_indices);
	        }
	      }
	    }
	    break;
	  case TridimPoolingParameter_PoolMethod_AVE:
	    // The main loop0, 1
	    for (int n = 0; n < top[0]->shape(0); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_ - pad_;
	              int wstart = pw * stride_ - pad_;
	              int lstart = pl * stride_d_ - pad_d_;
	              int hend = min(hstart + kernel_size_, height_ + pad_);
	              int wend = min(wstart + kernel_size_, width_ + pad_);
	              int lend = min(lstart + kernel_d_, length_ + pad_d_);
	              int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
	              hstart = max(hstart, 0);
	              wstart = max(wstart, 0);
	              lstart = max(lstart, 0);
	              hend = min(hend, height_);
	              wend = min(wend, width_);
	              lend = min(lend, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    bottom_diff[(l * height_ + h) * width_ + w] +=
	                      top_diff[(pl * pooled_height_ + ph) * pooled_width_ + pw] / pool_size;
	                  }
	                }
	              }
	            }
	          }
	    	}
	        // offset
	        bottom_diff += bottom[0]->offset(offset_indices);
	        top_diff += top[0]->offset(offset_indices);
	      }
	    }
	    break;
	  case TridimPoolingParameter_PoolMethod_SUM:
	    for (int n = 0; n < top[0]->shape(0); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_ - pad_;
	              int wstart = pw * stride_ - pad_;
	              int lstart = pl * stride_d_ - pad_d_;
	              int hend = min(hstart + kernel_size_, height_ + pad_);
	              int wend = min(wstart + kernel_size_, width_ + pad_);
	              int lend = min(lstart + kernel_d_, length_ + pad_d_);
	              int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
	              hstart = max(hstart, 0);
	              wstart = max(wstart, 0);
	              lstart = max(lstart, 0);
	              hend = min(hend, height_);
	              wend = min(wend, width_);
	              lend = min(lend, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    // bottom_diff[(l * height_ + h) * width_ + w] += top_diff[(pl * pooled_height_ + ph) * pooled_width_ + pw] / pool_size;
	                    bottom_diff[(l * height_ + h) * width_ + w] += top_diff[(pl * pooled_height_ + ph) * pooled_width_ + pw];
	                  }
	                }
	              }
	            }
	          }
	    	}
	        // offset
	        bottom_diff += bottom[0]->offset(offset_indices);
	        top_diff += top[0]->offset(offset_indices);
	      }
	    }
	  case TridimPoolingParameter_PoolMethod_STOCHASTIC:
	    NOT_IMPLEMENTED;
	    break;
	  default:
	    LOG(FATAL) << "Unknown pooling method.";
	  }

}

#ifdef CPU_ONLY
STUB_GPU(TridimPoolingLayer);
#endif

INSTANTIATE_CLASS(TridimPoolingLayer);
REGISTER_LAYER_CLASS(TridimPooling);

}  // namespace caffe
