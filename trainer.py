# coding: utf-8
import os, psutil, re
import tensorflow as tf
import tensorflow.keras as keras
from cnnkit import print_model_summary, optimizer_factory, load_weights_from, save_model_to
from centernet.loss import CombineLoss
from centernet.dataholder import DataHolder
from centernet.network import create_centernet, backbone_factory
from pymagic import logI, print_dict, is_file
from tqdm import tqdm


@tf.function
def train_one_batch(model, x, y, optimizer, loss_func):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        hmap_loss, offset_loss, size_loss, total_loss = loss_func(y_pred=pred, y_true=y)
        model_loss = tf.reduce_sum(model.losses)
        total_loss += model_loss
    grad = tape.gradient(target=total_loss, sources=model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))
    return hmap_loss, offset_loss, size_loss, total_loss

@tf.function
def eval_one_batch(model, x, y, loss_func):
    pred = model(x, training=False)
    hmap_loss, offset_loss, size_loss, total_loss = loss_func(y_pred=pred, y_true=y)
    return hmap_loss, offset_loss, size_loss, total_loss


class CenterNetTrainer:
    def __init__(self, config, model: keras.Model, optimizer, pretrained):
        super().__init__()
        self.model = model
        if pretrained and is_file(pretrained):
            load_weights_from(self.model, pretrained)
        self.optimizer = optimizer
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.num_classes = config.num_classes
        self.downsample_ratio = config.downsample_ratio
        self.loss_func = CombineLoss(config.num_classes,
                                     hmap_weight=config.hmap_loss_weight,
                                     off_weight=config.offset_loss_weight,
                                     size_weight=config.size_loss_weight)
        m = re.search("G([0-9]+)\\(", pretrained)
        if m:
            self.startG = int(m.groups()[0])
        else:
            self.startG = 0

    def _make_generator(self, dataholder_train, dataholder_valid) -> tuple:
        types = (tf.float32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        gen_train = tf.data.Dataset.from_generator(dataholder_train.generate_one, types)
        gen_valid = tf.data.Dataset.from_generator(dataholder_valid.generate_one, types)
        gen_train = gen_train.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        gen_valid = gen_valid.batch(self.batch_size, drop_remainder=True)
        return gen_train, gen_valid

    def run(self, max_epoches, dataholder_train, dataholder_valid, save_model_dir, save_name_prefix):
        logI("heatmap size:", self.input_size // self.downsample_ratio)

        for epoch in range(1, max_epoches + 1):
            mean_hmap_loss = keras.metrics.Mean()
            mean_offset_loss = keras.metrics.Mean()
            mean_size_loss = keras.metrics.Mean()
            mean_train_loss = keras.metrics.Mean()

            gen_train, gen_valid = self._make_generator(dataholder_train, dataholder_valid)

            pbar = tqdm(total=dataholder_train.num_batches, desc=F"Train {epoch}/{max_epoches}", mininterval=0.5)
            for x, y in gen_train:
                hmap_loss, offset_loss, size_loss, total_loss = train_one_batch(self.model, x, y, self.optimizer, self.loss_func)
                mean_hmap_loss.update_state(hmap_loss)
                mean_offset_loss.update_state(offset_loss)
                mean_size_loss.update_state(size_loss)
                mean_train_loss.update_state(total_loss)
                pbar.set_postfix(hmap=mean_hmap_loss.result().numpy(),
                                 offset=mean_offset_loss.result().numpy(),
                                 size=mean_size_loss.result().numpy(),
                                 total=mean_train_loss.result().numpy(),
                                 LR=self.optimizer.lr.numpy())
                self.optimizer.lr = max(1e-6, self.optimizer.lr * 0.9999)
                pbar.update(1)
            pbar.close()

            mean_hmap_loss = keras.metrics.Mean()
            mean_offset_loss = keras.metrics.Mean()
            mean_size_loss = keras.metrics.Mean()
            mean_valid_loss = keras.metrics.Mean()

            pbar = tqdm(total=dataholder_valid.num_batches, desc=F"Valid {epoch}/{max_epoches}", mininterval=0.5)
            for x, y in gen_valid:
                hmap_loss, offset_loss, size_loss, total_loss = eval_one_batch(self.model, x, y, self.loss_func)
                mean_hmap_loss.update_state(hmap_loss)
                mean_offset_loss.update_state(offset_loss)
                mean_size_loss.update_state(size_loss)
                mean_valid_loss.update_state(total_loss)
                pbar.set_postfix(hmap=mean_hmap_loss.result().numpy(),
                                 offset=mean_offset_loss.result().numpy(),
                                 size=mean_size_loss.result().numpy(),
                                 total=mean_valid_loss.result().numpy(),
                                 LR=self.optimizer.lr.numpy())
                self.optimizer.lr = max(1e-6, self.optimizer.lr * 0.9999)
                pbar.update(1)
            pbar.close()

            # save model
            train_loss = mean_train_loss.result().numpy()
            valid_loss = mean_valid_loss.result().numpy()
            logI("Epoch-%d final: train_loss=%.4f valid_loss=%.4f" % (epoch, train_loss, valid_loss))
            name = "%s-G%02d(%.4f_%.4f).h5" % (save_name_prefix, epoch+self.startG, train_loss, valid_loss)
            save_model_to(self.model, path=F"{save_model_dir}/{name}", include_optimizer=False, only_weights=False)



def train_centernet(conf, pretrained):
    backbone = backbone_factory(name=conf.backbone,
                                input_shape=conf.input_shape,
                                **conf.backbone_args)

    model = create_centernet(backbone=backbone,
                             input_shape=conf.input_shape,
                             num_classes=conf.num_classes,
                             deconv_layers=conf.deconv_layers,
                             deconv_filters=conf.deconv_filters,
                             deconv_kernels=conf.deconv_kernels,
                             head_channels=conf.head_channels,
                             act_type=conf.act_type)

    print_model_summary(model, F"{conf.save_model_dir}/cn-{conf.backbone}")

    optimizer = optimizer_factory(**conf.optimizer)
    print_dict(optimizer.get_config())

    DataHolder.debug_mode = 0
    dataholder_train = DataHolder(train_list_files=conf.train_list_files,
                                  input_size=conf.input_size,
                                  num_classes=conf.num_classes,
                                  batch_size=conf.batch_size,
                                  max_boxes=conf.max_boxes,
                                  downsample_ratio=conf.downsample_ratio)
    dataholder_train.init()

    dataholder_valid = DataHolder(train_list_files=conf.valid_list_files,
                                  input_size=conf.input_size,
                                  num_classes=conf.num_classes,
                                  batch_size=conf.batch_size,
                                  max_boxes=conf.max_boxes,
                                  downsample_ratio=conf.downsample_ratio)
    dataholder_valid.init()

    rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    logI(F"{round(rss, 2)}GB memory used")

    save_name_prefix = "cn-" + conf.backbone

    trainer = CenterNetTrainer(conf, model, optimizer, pretrained)
    trainer.run(conf.max_epoches, 
                dataholder_train, 
                dataholder_valid, 
                conf.save_model_dir,
                save_name_prefix)

