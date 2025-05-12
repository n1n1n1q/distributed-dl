#include <driver_types.h>
#include <fstream>
#include <torch/torch.h>
#include <iostream>
#include <boost/mpi.hpp>

#include "../src/distributed.hpp"

class CIFAR10Dataset : public torch::data::Dataset<CIFAR10Dataset> {
  std::vector<torch::Tensor> images_;
  std::vector<uint8_t> labels_;

public:
  CIFAR10Dataset(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file");
    }
    constexpr auto image_size = 3 * 32 * 32;
    constexpr auto record_size = image_size + 1;

    while (file.peek() != EOF) {
      std::vector<uint8_t> buffer(record_size);
      file.read(reinterpret_cast<char *>(buffer.data()), record_size);

      uint8_t label = buffer[0];

      // if (label == 1 || label == 2 /* || label == 3 || label == 4*/) {
      labels_.push_back(buffer[0]);
      // std::cout << labels_.back() << " ";

      torch::Tensor image = torch::from_blob(buffer.data() + 1, {3, 32, 32}, torch::kUInt8).clone();
      images_.push_back(image);
      // }
    }
    std::cout << "Loaded" << labels_.size() << "images" << std::endl;
  }

  torch::data::Example<> get(size_t idx) override {
    auto image = images_[idx].to(torch::kFloat32).div(255.0);
    auto label = torch::tensor(labels_[idx], torch::kLong);

    return {image, label};
  }

  [[nodiscard]] torch::optional<size_t> size() const override { return images_.size(); }
};

struct ResNetBlockImpl : torch::nn::Module {
  int64_t stride;
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
  torch::nn::Sequential downsample{nullptr};


  ResNetBlockImpl(int64_t inplanes, int64_t planes, int64_t stride = 1) : stride{stride} {
    conv1 = register_module(
      "conv1",
      torch::nn::Conv2d(
        torch::nn::Conv2dOptions(
          inplanes,
          planes,
          3)
        .stride(stride)
        .padding(1)
        .bias(false)));

    bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
    conv2 = register_module("conv2",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3)
                              .stride(1)
                              .padding(1)
                              .bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));

    // Define downsample if needed (when stride != 1 or channel sizes differ).
    if (stride != 1 || inplanes != planes) {
      downsample = register_module("downsample", torch::nn::Sequential(
                                     torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1)
                                       .stride(stride)
                                       .bias(false)),
                                     torch::nn::BatchNorm2d(planes)));
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto identity = x;
    auto out = conv1->forward(identity);
    out = bn1->forward(out);
    out = conv2->forward(out);
    out = bn2->forward(out);
    if (downsample) {
      identity = downsample->forward(x);
    }

    out += identity;

    return relu(out);
  }
};

TORCH_MODULE(ResNetBlock);

struct ResNetImpl : torch::nn::Module {
  int64_t inplanes = 64;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
  torch::nn::Linear fc{nullptr};


  ResNetImpl(int64_t num_classes = 10) {
    conv1 = register_module("conv1",
                            torch::nn::Conv2d(
                              torch::nn::Conv2dOptions(3, inplanes, 3).stride(1).padding(1).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(inplanes));

    layer1 = register_module("layer1", _make_layer(64, 2, 1));
    layer2 = register_module("layer2", _make_layer(128, 2, 2));
    layer3 = register_module("layer3", _make_layer(256, 2, 2));
    layer4 = register_module("layer4", _make_layer(512, 2, 2));

    fc = register_module("fc", torch::nn::Linear(512, num_classes));
  }

  torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride) {
    torch::nn::Sequential layers;
    layers->push_back(ResNetBlock(inplanes, planes, stride));
    inplanes = planes;
    for (int i = 1; i < blocks; i++) {
      layers->push_back(ResNetBlock(inplanes, planes));
    }
    return layers;
  }

  torch::Tensor forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = relu(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = adaptive_avg_pool2d(x, {1, 1});
    x = x.view({x.size(0), -1});
    x = fc->forward(x);
    return x;
  }
};

TORCH_MODULE(ResNet);


int main(int argc, char **argv) {
  bool is_cuda = true;
  DistributedTrainer dt(argc, argv, is_cuda);

  torch::Device device{torch::kCPU};
  if (is_cuda) {
    device = torch::Device(torch::kCUDA);
  }

  auto numranks = dt.size();
  auto rank = dt.rank();

  auto dataset = CIFAR10Dataset("./cifar-10-batches-bin/data_batch_1.bin")
      .map(torch::data::transforms::Normalize(0.5, 0.5))
      .map(torch::data::transforms::Stack<>());

  auto dist_data_sampler =
      torch::data::samplers::DistributedRandomSampler(dataset.size().value(), numranks, rank, false);

  auto num_train_samples_per_proc = dataset.size().value() / numranks;
  auto total_batch_size = 64;
  auto batch_size_per_proc = total_batch_size / numranks;

  auto data_loader = make_data_loader(std::move(dataset), dist_data_sampler, batch_size_per_proc);
  torch::manual_seed(0);


  ResNet model{10};
  model->to(device);

  torch::nn::CrossEntropyLoss loss_fn;
  torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));


  constexpr size_t num_epochs = 10;
  for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    model->train();
    size_t num_correct = 0;

    for (auto &batch: *data_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      optimizer.zero_grad();

      auto output = model->forward(data);


      auto loss = loss_fn(output, target);

      loss.backward();

      dt.synchronize_batch_norms(*model);
      dt.sync_train_step(model);

      optimizer.step();

      auto guess = output.argmax(1);
      num_correct += torch::sum(guess.eq_(target)).item<int64_t>();
    }
    auto accuracy = dt.sync_precision(100 * num_correct / num_train_samples_per_proc);

    std::cout << "Accuracy in rank " << rank << " in epoch " << epoch << " - "
        << accuracy << std::endl;
  }

  if (rank != 0)return 0;

  std::vector<at::Tensor> grads;

  for (auto &param: model->parameters()) {
    grads.push_back(param.grad().view({-1}));
  }

  auto meow = cat(grads); // gradients

  auto params_vec = model->parameters();
  torch::save(params_vec, "params_resnet.torch");

  return 0;
}
