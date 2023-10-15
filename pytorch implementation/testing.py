import torch


# add Gaussian noise to model's weights, in-place and magnitude-aware

def add_weight_noise_(
        model,
        stddev=0.0
):

    # get model weights as an Ordered Dictionary
    d = model.state_dict()

    # add noise to each Tensor (magnitude-aware)
    for key in d:
        noise = torch.randn_like(d[key]) * stddev
        d[key] = d[key] * (1 + noise)

    # load updated weights into the model
    model.load_state_dict(d)


# compute local energy to estimate flatness of loss landscape.

def estimate_flatness(
        model,
        loss_fn,
        val_data,
        noise,
        iters,
        out,                # write outputs here
        epoch,
        val_frequency,
        chkpt,              # save model checkpoint here (path with .pt extension)
):
    model.eval()  # eval mode
    n = epoch // val_frequency

    # save current weights
    torch.save(model.state_dict(), chkpt)

    with torch.no_grad():

        # repeatedly perturb model weights starting from initial configuration
        for i in range(iters):
            for s, stddev in enumerate(noise):
                # add noise to model weights
                add_weight_noise_(model, stddev)

                # compute loss with perturbed weights
                loss = loss_fn(model(val_data), val_data.flatten(start_dim=1))
                out[n * iters + i][s] = loss

                # reset weights to their initial values
                model.load_state_dict(torch.load(chkpt))

            # write current epoch
            out[n * iters + i][len(noise)] = epoch

    return out


# estimate denoising capabilities of model by feeding it noisy inputs and computing RE.

def estimate_denoising(
        model,
        loss_fn,
        val_data,
        noise,
        iters,
        out,                # write outputs here
        epoch,
        val_frequency,
):
    model.eval()  # eval mode
    n = epoch // val_frequency

    with torch.no_grad():

        # repeatedly denoise noisy inputs
        for i in range(iters):
            for s, stddev in enumerate(noise):

                # add noise
                noisy = val_data + torch.randn_like(val_data) * stddev

                # compute denoising RE
                loss = loss_fn(model(noisy), val_data.flatten(start_dim=1))
                out[n * iters + i][s] = loss

            # write current epoch
            out[n * iters + i][len(noise)] = epoch

    return out