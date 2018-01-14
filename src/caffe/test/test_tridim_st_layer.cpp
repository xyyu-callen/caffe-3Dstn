#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/tridim_st_layer.hpp"
#include "caffe/filler.hpp"
//#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "fstream"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class HardTridimSpatialTransformerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  HardTridimSpatialTransformerLayerTest() {
  	vector<int> U_shape(5);
  	U_shape[0] = 1;
  	U_shape[1] = 3;
  	U_shape[2] = 18;
  	U_shape[3] = 11;
  	U_shape[4] = 11;
  	vector<int> theta_shape(5);
  	theta_shape[0] = 1;
  	theta_shape[1] = 3;
  	theta_shape[2] = 4;
  	theta_shape[3] = 1;
  	theta_shape[4] = 1;
  	vector<int> V_shape(5);
  	V_shape[0] = 1;
  	V_shape[1] = 3;
  	V_shape[2] = 18;
  	V_shape[3] = 11;
  	V_shape[4] = 11;
  	blob_U_ = new Blob<Dtype>(U_shape);
  	blob_theta_ = new Blob<Dtype>(theta_shape);
  	blob_V_ = new Blob<Dtype>(V_shape);

  	FillerParameter filler_param;
  	GaussianFiller<Dtype> filler(filler_param);
  	filler.Fill(this->blob_U_);
  	filler.Fill(this->blob_theta_);

  	vector<int> shape_theta(2);
  	shape_theta[0] = 1; shape_theta[1] = 12;
  	blob_theta_->Reshape(shape_theta);

  	blob_bottom_vec_.push_back(blob_U_);
  	blob_bottom_vec_.push_back(blob_theta_);
  	blob_top_vec_.push_back(blob_V_);
  }
  virtual ~HardTridimSpatialTransformerLayerTest() { delete blob_V_; delete blob_theta_; delete blob_U_; }
  Blob<Dtype>* blob_U_;
  Blob<Dtype>* blob_theta_;
  Blob<Dtype>* blob_V_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HardTridimSpatialTransformerLayerTest, TestDtypesAndDevices);

TYPED_TEST(HardTridimSpatialTransformerLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	  	// reshape theta to have full 6 dimension
	  	vector<int> shape_theta(2);
		shape_theta[0] = 1; shape_theta[1] = 12;
		this->blob_theta_->Reshape(shape_theta);

		// fill random variables for theta
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_theta_);

		// set layer_param
		LayerParameter layer_param;
		TridimSpatialTransformerParameter *st_param = layer_param.mutable_tridim_st_param();
		st_param->set_output_h(11);
		st_param->set_output_w(11);
		st_param->set_output_d(11);

		// begin to check
		TridimSpatialTransformerLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-6, 1e-6);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(HardTridimSpatialTransformerLayerTest, TestGradientWithPreDefinedTheta) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

		// reshape theta to have only 2 dimensions
		vector<int> shape_theta(2);
		shape_theta[0] = 1; shape_theta[1] = 3;
		this->blob_theta_->Reshape(shape_theta);

		// fill random variables for theta
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_theta_);

		// set layer_param
		LayerParameter layer_param;
		TridimSpatialTransformerParameter *st_param = layer_param.mutable_tridim_st_param();
		st_param->set_output_h(11);
		st_param->set_output_w(11);
		st_param->set_output_d(11);

		st_param->set_theta_1_1(0.5);
		st_param->set_theta_1_2(0);
		st_param->set_theta_1_3(0);
		st_param->set_theta_2_1(0);
		st_param->set_theta_2_2(0.5);
		st_param->set_theta_2_3(0);
		st_param->set_theta_3_1(0);
		st_param->set_theta_3_2(0);
		st_param->set_theta_3_3(0.5);

		// begin to check
		TridimSpatialTransformerLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-6, 1e-6);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

