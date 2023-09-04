# coding: utf-8
import os, psutil, re
import tensorflow as tf
import tensorflow.keras as keras
from tfcnnkit import printModelSummary, OptimizerFactory, loadWeightsFrom, saveModelTo
from centernet.loss import CenterNetLoss
from centernet.dataholder import DataHolder
from centernet.network import createCenterNet, backboneFactory
from pymagic import logI, print_dict, isFile, redirectLoggingStream
from tqdm import tqdm


@tf.function
def trainOneBatch(model, x, y, optimizer, loss_func):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        hmap_loss, reg_loss, size_loss, total_loss = loss_func(y_pred=pred, y_true=y)
        model_loss = tf.reduce_sum(model.losses)
        total_loss += 0.1 * model_loss
    grad = tape.gradient(target=total_loss, sources=model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grad, model.trainable_variables))
    return hmap_loss, reg_loss, size_loss, total_loss

@tf.function
def evalOneBatch(model, x, y, loss_func):
    pred = model(x, training=False)
    hmap_loss, reg_loss, size_loss, total_loss = loss_func(y_pred=pred, y_true=y)
    return hmap_loss, reg_loss, size_loss, total_loss


class CenterNetTrainer:
    def __init__(self, config, model: keras.Model, optimizer, pretrained):
        super().__init__()
        self.model = model
        if pretrained and isFile(pretrained):
            loadWeightsFrom(self.model, pretrained)
        self.model.compile(optimizer)
        self.optimizer = optimizer
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.num_classes = config.num_classes
        self.loss_func = CenterNetLoss(config.num_classes,
                                       hmap_weight=config.hmap_loss_weight,
                                       reg_weight=config.reg_loss_weight,
                                       size_weight=config.size_loss_weight)
        m = re.search("G([0-9]+)\\(", pretrained)
        if m:
            self.startG = int(m.groups()[0])
        else:
            self.startG = 0

    def _makeGenerator(self, dataholder_train, dataholder_valid) -> tuple:
        types = (tf.float32, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        gen_train = tf.data.Dataset.from_generator(dataholder_train.generateOne, types)
        gen_valid = tf.data.Dataset.from_generator(dataholder_valid.generateOne, types)
        gen_train = gen_train.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        gen_valid = gen_valid.batch(self.batch_size, drop_remainder=True)
        return gen_train, gen_valid

    def run(self, max_epoches, dataholder_train, dataholder_valid, save_model_dir, save_name_prefix):
        redirectLoggingStream(F"{save_model_dir}/training.log", mode="w")
        
        for epoch in range(1, max_epoches + 1):
            mean_hmap_loss = keras.metrics.Mean()
            mean_reg_loss = keras.metrics.Mean()
            mean_size_loss = keras.metrics.Mean()
            mean_train_loss = keras.metrics.Mean()

            gen_train, gen_valid = self._makeGenerator(dataholder_train, dataholder_valid)

            pbar = tqdm(total=dataholder_train.num_batches, desc=F"Train {epoch}/{max_epoches}", mininterval=0.5)
            for x, y in gen_train:
                hmap_loss, reg_loss, size_loss, total_loss = trainOneBatch(self.model, x, y, self.optimizer, self.loss_func)
                mean_hmap_loss.update_state(hmap_loss)
                mean_reg_loss.update_state(reg_loss)
                mean_size_loss.update_state(size_loss)
                mean_train_loss.update_state(total_loss)
                pbar.set_postfix(hmap=mean_hmap_loss.result().numpy(),
                                 reg=mean_reg_loss.result().numpy(),
                                 size=mean_size_loss.result().numpy(),
                                 total=mean_train_loss.result().numpy(),
                                 LR=self.optimizer.lr.numpy())
                self.optimizer.lr = max(1e-6, self.optimizer.lr * 0.9999)
                pbar.update(1)
            pbar.close()

            mean_hmap_loss = keras.metrics.Mean()
            mean_reg_loss = keras.metrics.Mean()
            mean_size_loss = keras.metrics.Mean()
            mean_valid_loss = keras.metrics.Mean()

            pbar = tqdm(total=dataholder_valid.num_batches, desc=F"Valid {epoch}/{max_epoches}", mininterval=0.5)
            for x, y in gen_valid:
                hmap_loss, reg_loss, size_loss, total_loss = evalOneBatch(self.model, x, y, self.loss_func)
                mean_hmap_loss.update_state(hmap_loss)
                mean_reg_loss.update_state(reg_loss)
                mean_size_loss.update_state(size_loss)
                mean_valid_loss.update_state(total_loss)
                pbar.set_postfix(hmap=mean_hmap_loss.result().numpy(),
                                 reg=mean_reg_loss.result().numpy(),
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
            saveModelTo(self.model, path=F"{save_model_dir}/{name}", include_optimizer=False, only_weights=False)



def trainCenterNet(conf, pretrained):
    backbone = backboneFactory(name=conf.backbone,
                               input_shape=conf.input_shape,
                               **conf.backbone_args)

    model = createCenterNet(backbone=backbone,
                            input_shape=conf.input_shape,
                            num_classes=conf.num_classes,
                            deconv_filters=conf.deconv_filters,
                            head_channels=conf.head_channels,
                            act_type=conf.act_type)

    printModelSummary(model, F"{conf.save_model_dir}/cn-{conf.backbone}")

    optimizer = OptimizerFactory(**conf.optimizer)
    print_dict(optimizer.get_config())

    DataHolder.debug_mode = 0
    dataholder_train = DataHolder(train_list_files=conf.train_list_files,
                                  input_size=conf.input_size,
                                  num_classes=conf.num_classes,
                                  batch_size=conf.batch_size,
                                  max_boxes=conf.max_boxes)

    dataholder_valid = DataHolder(train_list_files=conf.valid_list_files,
                                  input_size=conf.input_size,
                                  num_classes=conf.num_classes,
                                  batch_size=conf.batch_size,
                                  max_boxes=conf.max_boxes).cache()

    rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    logI(F"{round(rss, 2)}GB memory used")

    save_name_prefix = "cn-" + conf.backbone

    trainer = CenterNetTrainer(conf, model, optimizer, pretrained)
    trainer.run(conf.max_epoches, 
                dataholder_train, 
                dataholder_valid, 
                conf.save_model_dir,
                save_name_prefix)


