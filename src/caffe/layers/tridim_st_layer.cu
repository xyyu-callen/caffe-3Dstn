#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/layers/tridim_st_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_value_to_constant(const int nthreads, Dtype value, int size, 
	int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size + i] = value;
	}
}

template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k, 
	const Dtype* src, int size_dst, int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size_dst + i] = src[index * size_src + k];
	}
}

template <typename Dtype>
__global__ void TridimSpatialTransformerForwardGPU(const int nthreads, int N, int C,
		int output_D_, int output_H_, int output_W_, int D, int H, int W,
		const Dtype* input_grid_data, const Dtype* U, Dtype* V) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int d = (index / (output_W_ * output_H_)) % output_D_;
		const int j = (index / (output_D_ * output_W_ * output_H_)) % C;
		const int i = index / (output_D_ * output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_D_ * output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * output_H_ *d + output_W_ * s + t;

	  	const Dtype pz = coordinates[row_idx * 3];
	  	const Dtype px = coordinates[row_idx * 3 + 1];
	  	const Dtype py = coordinates[row_idx * 3 + 2];

	  	const int V_offset = index;

	  	V[V_offset] = (Dtype)0.;

	  	const Dtype z = (pz + 1) / 3 * D;
	  	const Dtype x = (px + 1) / 3 * H;
	  	const Dtype y = (py + 1) / 3 * W;

	  	int l, m, n; Dtype w;
	  	const Dtype* pic = U + i * (C * D * H * W) + j * (D * H * W);

	  	l = floor(z); m = floor(x); n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (x - m)) * (1 - (y - n));
	  		V[V_offset] += w * pic[l * W * H + m * W + n];
	  	}

	  	l = floor(z); m = floor(x) + 1; n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (m - x)) * (1 - (y - n));
	  		V[V_offset] += w * pic[l * W * H + m * W + n];
	  	}

	  	l = floor(z); m = floor(x); n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (x - m)) * (1 - (n - y));
	  		V[V_offset] += w * pic[l * W * H + m * W + n];
	  	}

	  	l = floor(z); m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (m - x)) * (1 - (n - y));
	  		V[V_offset] += w * pic[l * H * W + m * W + n];
	  	}

	  	l = floor(z + 1); m = floor(x); n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (x - m)) * (1 - (y - n));
	  		V[V_offset] += w * pic[l * W * H + m * W + n];
	  	}

	  	l = floor(z + 1); m = floor(x) + 1; n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (m - x)) * (1 - (y - n));
	  		V[V_offset] += w * pic[l * W * H + m * W + n];
	  	}

	  	l = floor(z + 1); m = floor(x); n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (x - m)) * (1 - (n - y));
	  		V[V_offset] += w * pic[l * W * H + m * W + n];
	  	}

	  	l = floor(z + 1); m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (m - x)) * (1 - (n - y));
	  		V[V_offset] += w * pic[l * H * W + m * W + n];
	  	}
  }
}

template <typename Dtype>
void TridimSpatialTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "TridimSpatialTransformerLayer::Forward_gpu::\t";

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	const Dtype* output_grid_data = output_grid.gpu_data();
	
	Dtype* full_theta_data = full_theta.mutable_gpu_data();
	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);
	
	// compute full_theta
	int k = 0; 
	const int num_threads = N;
	for(int i=0; i<12; ++i) {
		if(is_pre_defined_theta[i]) {
			set_value_to_constant<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>( 
				num_threads, pre_defined_theta[i], 12, i, full_theta_data);
			//std::cout << "Setting value " << pre_defined_theta[i] << " to "<< i << 
			//	"/12 of full_theta_data" << std::endl;
		} else {
			copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(num_threads, 
				12 - pre_defined_count, k, theta, 12, i, full_theta_data);
			//std::cout << "Copying " << k << "/" << 12 - pre_defined_count << " of theta to " 
			//	<< i << "/12 of full_theta_data" << std::endl;
			++ k;
		}
	}

	// compute out input_grid_data
	for(int i = 0; i < N; ++i) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_D_ * output_H_ * output_W_, 3, 4, (Dtype)1.,
				output_grid_data, full_theta_data + 12 * i, (Dtype)0.,
				input_grid_data + (output_D_ * output_H_ * output_W_ * 3) * i);
	}

	const int nthreads = N * C * output_D_ * output_H_ * output_W_;

	TridimSpatialTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_D_, output_H_, output_W_, D, H, W, input_grid_data, U, V);
}

template <typename Dtype>
__global__ void TridimSpatialTransformerBackwardGPU_dTheta(const int nthreads, int C,
		int output_D_, int output_H_, int output_W_, int D, int H, int W,
		const Dtype* input_grid_data, const Dtype* dV_array, const Dtype* U_array,  
		Dtype* dTheta_tmp_diff) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int d = (index / (output_W_ * output_H_)) % output_D_;
		const int j = (index / (output_D_ * output_W_ * output_H_)) % C;
		const int i = index / (output_D_ * output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_D_ * output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * output_H_ * d + output_W_ * s + t;

		const Dtype pz = coordinates[row_idx * 3];
		const Dtype px = coordinates[row_idx * 3 + 1];
		const Dtype py = coordinates[row_idx * 3 + 2];
		
		Dtype delta_dpz = (Dtype)0.;
		Dtype delta_dpx = (Dtype)0.;
		Dtype delta_dpy = (Dtype)0.;

		const Dtype z = (pz + 1) / 3 * D;
		const Dtype x = (px + 1) / 3 * H;
		const Dtype y = (py + 1) / 3 * W;
		const int dV_offset = index;
		const Dtype dV = dV_array[dV_offset];

		int l, m, n; 
		const Dtype* U = U_array + i * (C * D * H * W) + j * (D * H * W);

		// left-bottom neighbor
		l = floor(z); m = floor(x); n = floor(y); 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (z - l)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * H * D / 3;
			delta_dpy -= (1 - (z - l)) * (1 - (x - m)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz -= (1 - (x - m)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * W * D / 3;
		}
		
		// left-top neighbor
		l = floor(z); m = floor(x); n = floor(y) + 1; 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (z - l)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * H * D / 3;
			delta_dpy += (1 - (z - l)) * (1 - (x - m)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz -= (1 - (x - m)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * W * D / 3;
		}

		// right-bottom neighbor
		l = floor(z); m = floor(x) + 1; n = floor(y); 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (z - l)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpy -= (1 - (z - l)) * (1 - (m - x)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz -= (1 - (m - x)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * W * D / 3;
		}
		
		// right-top neighbor
		l = floor(z); m = floor(x) + 1; n = floor(y) + 1; 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (z - l)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpy += (1 - (z - l)) * (1 - (m - x)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz -= (1 - (m - x)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * W * D / 3;	
		}

		// left-bottom neighbor
		l = floor(z + 1); m = floor(x); n = floor(y); 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (l - z)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * H * D / 3;
			delta_dpy -= (1 - (l - z)) * (1 - (x - m)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz += (1 - (x - m)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * W * D / 3;
		}
		
		// left-top neighbor
		l = floor(z + 1); m = floor(x); n = floor(y) + 1; 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (z - l)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * H * D / 3;
			delta_dpy += (1 - (z - l)) * (1 - (x - m)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz += (1 - (x - m)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * W * D / 3;
		}

		// right-bottom neighbor
		l = floor(z + 1); m = floor(x) + 1; n = floor(y); 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (z - l)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpy -= (1 - (z - l)) * (1 - (m - x)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz += (1 - (m - x)) * (1 - (y - n)) * U[l * W * H + m * W + n] * dV * W * D / 3;
		}
		
		// right-top neighbor
		l = floor(z + 1); m = floor(x) + 1; n = floor(y) + 1; 
		if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (z - l)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpy += (1 - (z - l)) * (1 - (m - x)) * U[l * W * H + m * W + n] * dV * W * D / 3;
			delta_dpz += (1 - (m - x)) * (1 - (n - y)) * U[l * W * H + m * W + n] * dV * W * D / 3;	
		}
		
		int idx = j * (output_D_ * output_H_ * output_W_) + d * output_W_ * output_H_ + s * output_W_ + t;
		
		dTheta_tmp_diff[(12 * i) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpx * (s * 1.0 / output_H_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 1) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpx * (t * 1.0 / output_W_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 2) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpx * (d * 1.0 / output_D_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 3) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpx;
		dTheta_tmp_diff[(12 * i + 4) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpy * (s * 1.0 / output_H_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 5) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpy * (t * 1.0 / output_W_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 6) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpy * (d * 1.0 / output_D_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 7) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpy;
		dTheta_tmp_diff[(12 * i + 8) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpz * (s * 1.0 / output_H_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 9) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpz * (t * 1.0 / output_W_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 10) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpz * (d * 1.0 / output_D_ * 2 - 1);
		dTheta_tmp_diff[(12 * i + 11) * (output_D_ * output_H_ * output_W_ * C) + idx] += delta_dpz;
	}
}

template <typename Dtype>
__global__ void TridimSpatialTransformerBackwardGPU_dU(const int nthreads, const int C, 
	const int D, const int W,  const int H, const int output_D_, const int output_H_, const int output_W_, const Dtype* input_grid_data, const Dtype* dV, Dtype* dU) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int d = (index / (output_W_ * output_H_)) % output_D_;
		const int j = (index / (output_D_ * output_W_ * output_H_)) % C;
		const int i = index / (output_D_ * output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_D_ * output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * output_H_ * d + output_W_ * s + t;

	  	const Dtype pz = coordinates[row_idx * 3];
	  	const Dtype px = coordinates[row_idx * 3 + 1];
	  	const Dtype py = coordinates[row_idx * 3 + 2];

	  	const int V_offset = index;

	  	const Dtype z = (pz + 1) / 3 * D;
	  	const Dtype x = (px + 1) / 3 * H;
	  	const Dtype y = (py + 1) / 3 * W;

	  	int l, m, n; Dtype w;
	  	Dtype* pic = dU + i * (C * D * H * W) + j * (D * H * W);

	  	l = floor(z); m = floor(x); n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (x - m)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}

	  	l = floor(z); m = floor(x) + 1; n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (m - x)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}

	  	l = floor(z); m = floor(x); n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (x - m)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}

	  	l = floor(z); m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (z - l)) * (1 - (m - x)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}

	  	l = floor(z + 1); m = floor(x); n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (x - m)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}

	  	l = floor(z + 1); m = floor(x) + 1; n = floor(y); w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (m - x)) * (1 - (y - n));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}

	  	l = floor(z + 1); m = floor(x); n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (x - m)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}

	  	l = floor(z + 1); m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(l >= 0 && l < D && m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (l - z)) * (1 - (m - x)) * (1 - (n - y));
			caffe_gpu_atomic_add(w * dV[V_offset], pic + (l * W * H + m * W + n));
	  	}
	}
}

template <typename Dtype>
void TridimSpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "TridimSpatialTransformerLayer::Backward_GPU::\t";

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* input_grid_data = input_grid.gpu_data();
	const Dtype* U = bottom[0]->gpu_data();

	Dtype* dFull_theta = full_theta.mutable_gpu_diff();
	Dtype* dTheta = bottom[1]->mutable_gpu_diff();
	Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();

	caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);

	const int nthreads = N * C * output_D_ * output_H_ * output_W_;

	TridimSpatialTransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_D_, output_H_, output_W_, D, H, W, input_grid_data,
					dV, U, dTheta_tmp_diff);

	Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
	caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, full_theta.count(), 1, output_D_ * output_H_ * output_W_ * C, 
			(Dtype)1., dTheta_tmp_diff, all_ones_2_data, (Dtype)0., dFull_theta);
			
	/*const Dtype* db_dFull_theta = full_theta.cpu_diff();
	for(int i=0; i<full_theta.count(); ++i) {
		std::cout << db_dFull_theta[i] << " ";
	}
	std::cout<<std::endl;*/
			
	int k = 0;
	const int num_threads = N;
	for(int i=0; i<12; ++i) {
		if(!is_pre_defined_theta[i]) {
			copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(num_threads, 
				12, i, dFull_theta, 12 - pre_defined_count, k, dTheta);
			//std::cout << "Copying " << i << "/12 of dFull_theta to " << k << "/" << 
			//	12 - pre_defined_count << " of dTheta" << std::endl;
			++ k;
		}
	}
	
	/*const Dtype* db_dtheta = bottom[1]->cpu_diff();
	for(int i=0; i<bottom[1]->count(); ++i) {
		std::cout << db_dtheta[i] << " ";
	}
	std::cout<<std::endl;*/
			
	if(to_compute_dU_) {
		Dtype* dU = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
		const int nthreads = N * C * output_H_ * output_W_;
		TridimSpatialTransformerBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, D, W, H, output_D_, output_H_, output_W_, input_grid_data, dV, dU);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(TridimSpatialTransformerLayer);

}	// namespace caffe

