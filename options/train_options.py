from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from one checkpoint')

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=400, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
        parser.add_argument('--patch_size', type=int, default=128, help='patch size of images for cropping. 0 means no cropping')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # val
        parser.add_argument('--val_interval', type=int, default=10, help='the interval between the output images in the validation set')

        self.isTrain = True
        return parser
