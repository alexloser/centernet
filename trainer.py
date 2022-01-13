# coding: utf-8
import os, psutil
import tensorflow as tf
import tensorflow.keras as keras
from cnnkit import print_model_summary, optimizer_factory, load_weights_from, SaveNameCallback
from centernet.loss import CombineLoss
from centernet.dataholder import DataHolder
from centernet.network import create_centernet, backbone_factory
from pymagic import logI, real_path, print_dict, is_file
from tqdm import tqdm


@tf.function
def train_one_batch(model, x, y, optimizer, loss_func):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        hmap_loss, offset_loss, size_loss, total_loss = loss_func(y_pred=pred, y_true=y)
    grad = tape.gradient(target=total_loss, sources=model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))
    return hmap_loss, offset_loss, size_loss, total_loss


class CenterNetTrainer:
    def __init__(self, model:keras.Model, config):
        super().__init__()
        self.model = model
        self.conf = config

    def save(self, name):
        path = F"{self.conf.save_model_dir}/{name}"
        logI("Saved model:", real_path(path))
        self.model.save(filepath=path, save_format="h5", include_optimizer=False)

    def run(self, pretraind, save_name_callback):
        conf = self.conf
        logI("heatmap size:", conf.input_size // conf.downsampling_ratio)

        optimizer = optimizer_factory(**conf.optimizer)
        loss_func = CombineLoss(conf.num_classes,
                                hmap_weight=conf.hmap_loss_weight,
                                off_weight=conf.offset_loss_weight,
                                size_weight=conf.size_loss_weight)
        print_dict(optimizer.get_config())
        logI(F"loss: {loss_func}")

        mean_hmap_loss = keras.metrics.Mean()
        mean_offset_loss = keras.metrics.Mean()
        mean_size_loss = keras.metrics.Mean()
        mean_total_loss = keras.metrics.Mean()

        self.model.compile(optimizer=optimizer)
        if pretraind and is_file(pretraind):
            load_weights_from(self.model, pretraind)

        DataHolder.debug_mode = 0
        data_holder = DataHolder(train_list_files=conf.train_list_files,
                                 input_size = conf.input_size,
                                 num_classes=conf.num_classes,
                                 batch_size=conf.batch_size,
                                 max_boxes=conf.max_boxes,
                                 downsampling_ratio=conf.downsampling_ratio)
        data_holder.init()

        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        logI(F"{round(rss, 2)}GB memory used")

        for epoch in range(1, conf.max_epoches+1):
            mean_hmap_loss.reset_states()
            mean_offset_loss.reset_states()
            mean_size_loss.reset_states()
            mean_total_loss.reset_states()
            pbar = tqdm(total=data_holder.num_batches, desc=F"Epoch {epoch}/{conf.max_epoches}", mininterval=0.5)
            
            for x, y in data_holder.generate():
                hmap_loss, offset_loss, size_loss, total_loss = train_one_batch(self.model, x, y, optimizer, loss_func)
                mean_hmap_loss.update_state(hmap_loss)
                mean_offset_loss.update_state(offset_loss)
                mean_size_loss.update_state(size_loss)
                mean_total_loss.update_state(total_loss)
                pbar.set_postfix(hmap=mean_hmap_loss.result().numpy(),
                                 offset=mean_offset_loss.result().numpy(),
                                 size=mean_size_loss.result().numpy(),
                                 total=mean_total_loss.result().numpy(),
                                 LR=optimizer.lr.numpy())
                optimizer.lr = max(1e-6, optimizer.lr * 0.9999)
                pbar.update(1)
            pbar.close()

            name = save_name_callback(conf.input_size, epoch, loss=mean_total_loss.result().numpy())
            self.save(name)


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

    trainer = CenterNetTrainer(model, conf)
    trainer.run(pretrained, SaveNameCallback("cn", conf.backbone))

