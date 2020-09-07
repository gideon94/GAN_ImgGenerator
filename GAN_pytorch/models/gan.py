import torch
import torch.nn as nn
import torch.nn.parallel


class GanDiscriminator(nn.Module):
    
    def __init__(self, frame_size, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(GanDiscriminator, self).__init__()
        self.ngpu = ngpu
        assert frame_size % 16 == 0, "frame_size has to be a multiple of 16"
        main = nn.Sequential()
        main.add_module('initial_convolution_{0}_{1}'.format(nc, ndf)
                        , nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_reLu{0}'.format(ndf)
                        , nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = frame_size / 2, ndf
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra_layers_convolution_{0}{1}'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra_layers_batchnorm_{0}{1}'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra_layers_reLu{0}{1}'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_convolution_{0}{1}'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_batchnorm_{0}'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_reLu_{0}'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final_convolution{0}{1}'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)


class GanGenerator(nn.Module):

    def __init__(self, frame_size, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(GanGenerator, self).__init__()
        self.ngpu = ngpu
        assert frame_size % 16 == 0
        cngf, tframe_size = ngf // 2, 4
        while tframe_size != frame_size:
            cngf = cngf * 2
            tframe_size = tframe_size * 2

        main = nn.Sequential()
        main.add_module('initial_convolution_{0}{1}'.format(nz, cngf)
                        , nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_batchnorm_{0}'.format(cngf)
                        , nn.BatchNorm2d(cngf))
        main.add_module('initial_reLu_{0}'.format(cngf)
                        , nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < frame_size // 2:
            main.add_module('pyramid_convolution_{0}{1}'.format(cngf, cngf // 2)
                            , nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_batchnorm_{0}'.format(cngf // 2)
                            , nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_reLu_{0}'.format(cngf // 2)
                            , nn.ReLU(True))

            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra_layers_convolution_{0}{1}'.format(t, cngf)
                            , nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra_layers_batchnorm_{0}{1}'.format(t, cngf)
                            , nn.BatchNorm2d(cngf))
            main.add_module('extra_layers_reLu{0}{1}'.format(t, cngf)
                            , nn.ReLU(True))

        main.add_module('final_convolution_{0}{1}'.format(cngf, nc)
                        , nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final_reLu_{0}'.format(nc)
                        , nn.ReLU())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
