std::tuple<torch::Tensor, torch::Tensor> train_jit_module(
        std::string jit_model_pt,
        torch::Tensor x_train,
        torch::Tensor y_train,
        torch::Tensor x_val,
        int nepochs) {
    torch::manual_seed(SEED);
    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return std::make_tuple(torch::Tensor{},torch::Tensor{}) ;
    auto &net = module.value();
    net.train();

    auto loss_fn = torch::nn::MSELoss{};
    auto adam_optim = torch::optim::Adam{parameters(net), torch::optim::AdamOptions(0.005)}; 
    auto adagrad_optim = torch::optim::Adagrad{parameters(net), torch::optim::AdagradOptions(0.005)}; 
    auto adam_w_optim = torch::optim::AdamW{parameters(net), torch::optim::AdamWOptions(0.005)}; 
    auto rmsprop_optim = torch::optim::RMSprop{parameters(net), torch::optim::RMSpropOptions(0.005)}; 
    auto sgd_optim = torch::optim::SGD{parameters(net), torch::optim::SGDOptions(0.005)};
    
    torch::optim::Optimizer* optimizer = static_cast<torch::optim::Optimizer*>(&adam_optim);

    auto optim_preds = std::vector<at::Tensor>{};
    optim_preds.reserve(nepochs);

    for (int i = 0; i < nepochs; i++) {

        optimizer->zero_grad();
        auto output = net({x_train}).toTensor();
        auto loss = loss_fn(output, y_train);
        loss.backward();
        optimizer->step();

        optim_preds.push_back(net({x_val}).toTensor().detach());
    }
    optimizer->zero_grad();

    return std::make_tuple(flat_parameters(net, true), torch::stack(optim_preds));
}
