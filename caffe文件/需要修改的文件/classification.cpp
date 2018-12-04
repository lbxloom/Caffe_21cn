#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//#include <time.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces) //命名空间caffe
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {	//C++类Classifier，类的定义和实现都放在一个.cpp文件中，在main函数中使用到该类对象
 public:
  Classifier(const string& model_file,		//模型描述文件deploy.prototxt
             const string& trained_file,	//训练好的模型.caffemodel
             const string& mean_file,		//均值文件mean.binaryproto
             const string& label_file);		//标签文本labels.txt---类别号与类别名对照

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 2);	//分类函数，输入Mat图像数据和top N，默认为top 5，这里改成top 2

  ////////////////////新增
  std::vector<Prediction> Classify(const cv::Mat& img, int N, int task_index);	//重载分类函数，输入Mat图像数据、top N、多任务索引
  ///////////////////////															//任务索引为0时进行前向传播，返回任务0的预测结果；不为0则不需要再进行前向传播了，直接返回该任务索引的预测结果

 private:
  void SetMean(const string& mean_file);	//根据mean_file均值文件生成均值Mat图像mean_

  std::vector<float> Predict(const cv::Mat& img);

  ////////////////////新增
  std::vector<float> Predict(const cv::Mat& img, int task_index);	//重载预测函数，任务索引为0时进行前向传播，返回任务0的预测结果；不为0则不需要再进行前向传播了，直接返回该任务索引的预测结果
  ///////////////////////	

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;	//网络net_，caffe中成员变量都有后缀_，这里保持这种风格
  cv::Size input_geometry_;		//几何维度(width,height)
  int num_channels_;			//通道数
  cv::Mat mean_;				//均值Mat图像
  std::vector<string> labels_;	//标签（类别）名向量

  ///////新增
  std::vector<vector<string>> all_task_labels_;	//多任务所有标签名向量
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {	//重载的构造函数
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);	//CPU模式
#else
  Caffe::set_mode(Caffe::GPU);	//GPU模式
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));	//网络nte_设置网络模型结构及TEST模式
  net_->CopyTrainedLayersFrom(trained_file);	//网络net_设置训练好的网络模型

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";		//网络的输入Blob数目，只能有1个，即net_->input_blobs()[0]
//  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";	//网络的输出Blob数目	////////////////////////////////////////////////////////////////////////////////////////////////

  Blob<float>* input_layer = net_->input_blobs()[0];	//网络的输入Blob数组的首个Blob指针，Blob是四维数组(width_,height_,channels_,num_)
  num_channels_ = input_layer->channels();				//获取输入Blob（首个）的通道数
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());	//获取输入Blob（首个）的几何维度，cv::Size

  /* Load the binaryproto mean file. */
  SetMean(mean_file);	//根据mean_file均值文件，生成均值图像mean_

  /* Load labels. */
  std::ifstream labels(label_file.c_str());		//打开文件流labels，c_str()函数返回一个指向正规C字符串的指针, 内容与本string串（标签文本路径label_file）相同
  CHECK(labels) << "Unable to open labels file " << label_file;
  /*
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));			
	*/
  //读取多任务标签对照表
  int task_num;
  labels >> task_num;
  vector<int> label_num(task_num);
  int i, j, index;	//临时变量
  for (i = 0; i < task_num; i++)
	  labels >> label_num[i];
  for (i = 0; i < task_num; i++){
	  vector<string> my_labels(label_num[i]);
	  for (j = 0; j < label_num[i]; j++)
		  labels >> index >> my_labels[j];
	  all_task_labels_.push_back(my_labels);
  }
  
  /*
  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())	//标签文本中的类别数要和输出Blob的通道数（其实就是分的类别总数）一致	////////////////////////////////////////////////////////////////////////////////////////////////
    << "Number of labels is different from the output layer dimension.";
	*/
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {	//返回top N 的索引，二类问题中N=1
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {	//Classifier类成员函数，用于分类，返回top N 的预测类别
  std::vector<float> output = Predict(img);		//获取预测的概率float向量

  N = std::min<int>(labels_.size(), N);			//若N比标签类别总数还大，则取N为标签类别总数
  std::vector<int> maxN = Argmax(output, N);	//返回top N 的索引	
  std::vector<Prediction> predictions;			//预测结果（向量，top N 的类别名和概率值）
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));	//返回的预测结果为：top N 的类别名和概率值
  }

  return predictions;
}

//////////////////////新增的重载函数Classify
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N, int task_index)
{
	std::vector<float> output = Predict(img, task_index);		//获取预测的概率float向量

	N = std::min<int>(all_task_labels_[task_index].size(), N);			//若N比标签类别总数还大，则取N为标签类别总数
	std::vector<int> maxN = Argmax(output, N);	//返回top N 的索引	
	std::vector<Prediction> predictions;			//预测结果（向量，top N 的类别名和概率值）
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(all_task_labels_[task_index][idx], output[idx]));	//返回的预测结果为：top N 的类别名和概率值
	}

	return predictions;
}
////////////////////////////////

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);	//从mean.binaryproto中读取到blob_proto

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);	//从BlobProto到Blob
  CHECK_EQ(mean_blob.channels(), num_channels_)	//均值文件图像通道数要与输入图像通道数一致
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();	//读写访问cpu data，32-bit float BGR or grayscale
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);	//构造函数生成Mat
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();	//为了分离通道
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);	//通道融合,得到Mat图像数据mean

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean); //计算每个通道的均值，构成一个三维向量，例如ImageNet数据集的为104,117,123（R,G,B）
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);	//用三维向量初始化各通道生成一副Mat图像mean_
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);	//将输入Blob的内容转换成各通道的cv::Mat图像（向量）

  Preprocess(img, &input_channels);	//做颜色空间转换、缩放、去均值等预处理，返回通道分离后的各通道Mat

  net_->Forward();	//前向传播

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];	//得到前向传播后的输出Blob
  const float* begin = output_layer->cpu_data();		//输出Blob的数据首指针
  const float* end = begin + output_layer->channels();	//输出Blob的数据未指针
  return std::vector<float>(begin, end);				//返回预测概率结果向量
}

/////////////////新增的重载Predict
std::vector<float> Classifier::Predict(const cv::Mat& img,int task_index) {
	if (task_index == 0){
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);	//将输入Blob的内容转换成各通道的cv::Mat图像（向量）

		Preprocess(img, &input_channels);	//做颜色空间转换、缩放、去均值等预处理，返回通道分离后的各通道Mat

		net_->Forward();	//前向传播
	}
	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[task_index];	//得到前向传播后的输出Blob
	const float* begin = output_layer->cpu_data();		//输出Blob的数据首指针
	const float* end = begin + output_layer->channels();	//输出Blob的数据未指针
	return std::vector<float>(begin, end);				//返回预测概率结果向量
}
//////////////////////

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {	//将输入Blob的内容转换成cv::Mat图像数据
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {		//做颜色空间转换、缩放、去均值等预处理，返回通道分离后的各通道Mat
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)	//颜色转换
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)	//缩放
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);	//去均值

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);		//通道分离

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];	//模型描述文件deploy.prototxt
  string trained_file = argv[2];	//训练好的网络模型network.caffemodel
  string mean_file    = argv[3];	//均值文件mean.binaryproto
  string label_file   = argv[4];	//标签文件labels.txt---类别号与类别名对应表
  Classifier classifier(model_file, trained_file, mean_file, label_file);	//构造Classifier类对象，构造函数输入参数包括上面四个

  string file = argv[5];	//待分类测试的图像img.jpg

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  /*
  std::vector<Prediction> predictions = classifier.Classify(img);	//调用Classifier类对象的分类函数Classify，传入参数为cv::Mat图像数据，返回

  /* Print the top N predictions. */
  /*
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
  */

  ////////////////////////////新增，以支持多任务分类
  //clock_t time_start, time_end;
  //time_start = clock();
  int TASK_NUM = 4;	//多任务数
  int TOP_N = 1;	//top N
  for (int index = 0; index < TASK_NUM; index++)
  {
	  std::vector<Prediction> predictions = classifier.Classify(img, TOP_N, index);	//进行多任务分类，并传入任务索引(索引为index=0时进行前向传播，返回任务0的预测结果；索引为index！=0时，不需要在做前向传播，直接返回任务index的预测结果)
	  for (size_t i = 0; i < predictions.size(); ++i) {	//打印预测的top N
		  Prediction p = predictions[i];
		  std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			  << p.first << "\"" << std::endl;
	  }
  }
  //time_end = clock();
  //double duration = (double)(time_end - time_start) / CLOCKS_PER_SEC;
  //std::cout << "duaration:" << duration << std::endl;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
