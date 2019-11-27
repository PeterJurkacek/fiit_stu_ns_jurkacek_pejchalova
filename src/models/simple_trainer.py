from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import logging
import src.logger as logger
import src.data.load_dataset as loader
from src.utils import calculate_steps_per_epoch, timestamp
from src import config


def start(model, run_id=timestamp()):
    train_data = loader.load_train_dataset()
    validation_data = loader.load_test_dataset()
    compile(model)
    train(model=model, train_data=train_data, validation_data=validation_data, run_id=run_id)
    evaluate(run_id=run_id, test_data=validation_data, steps=loader.test_data_count)


def compile(model):
    logging.info(f"model.compile()")
    logging.info(f"loss:{config.loss}")
    logging.info(f"metrics:{config.metrics}")
    logging.info(f"optimizer:{config.optimizer}")
    logging.info(f"learning_rate:{config.optimizer.learning_rate}")

    model.compile(
        loss=config.loss,
        optimizer=config.optimizer,
        metrics=config.metrics,
    )
    return model


def train(model, train_data, validation_data, run_id):
    logging.info(f"model.fit()")
    logging.info(f"run_id: {run_id}")
    logging.info(f"epochs:{config.epochs}")
    steps_per_epoch = steps_per_epoch_train()
    validation_steps = steps_per_epoch_validate()
    logging.info(f"steps_per_epoch:{steps_per_epoch}")
    logging.info(f"validation_steps:{validation_steps}")

    history = model.fit(train_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=config.epochs,
                        validation_data=validation_data,
                        validation_steps=validation_steps,
                        callbacks=[logger.create_tensorboard_callback(run_id),
                                   logger.create_csv_logger_callback(run_id)])
    # Save the model
    model_path = logger.get_model_path(run_id)
    model.save(model_path)
    logging.info(f"Model {run_id} saved to model_path: {model_path}")


def evaluate(run_id, test_data, steps):
    # Recreate the exact same model, including its weights and the optimizer
    logging.info(f"model.evaluate()")
    logging.info(f"run_id: {run_id}")
    logging.info(f"test_data: {test_data}")
    logging.info(f"steps: {steps}")
    model_path = logger.get_model_path(run_id)
    model = tf.keras.models.load_model(model_path)
    logging.info(f"model was loaded from{model_path}")

    # Show the model architecture
    model.summary(print_fn=logging.info)
    model.evaluate(test_data,
                   steps=steps,
                   callbacks=[logger.create_tensorboard_callback(run_id=run_id)])


def steps_per_epoch_train():
    return calculate_steps_per_epoch(loader.train_data_count, loader.batch_size)


def steps_per_epoch_validate():
    return calculate_steps_per_epoch(loader.test_data_count, loader.batch_size)
