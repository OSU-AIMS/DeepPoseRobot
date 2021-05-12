
from robotpose.utils import setMemoryGrowth
from robotpose import Dataset
import tensorflow as tf

from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.models import DeepLabCut, StackedDenseNet, StackedHourglass, LEAP
from deepposekit.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from deepposekit.callbacks import Logger, ModelCheckpoint
from robotpose.paths import Paths as p
import os
import argparse
from robotpose.utils import workerCount

setMemoryGrowth()

def run(dataset, skeleton, model_type, batch_size, valid_size):

    ds = Dataset(dataset, skeleton)
    print("Dataset loaded")
    data_generator = DataGenerator(ds.deepposeds_path)
    print("Data Generator loaded")

    model_path = os.path.join(p().MODELS,os.path.basename(os.path.normpath(ds.deepposeds_path)).replace('.h5',f'_{model_type}.h5'))
    model_path = os.path.join(p().MODELS,f"{ds.name}__{ds.skele.name}__{model_type}.h5")

    if os.path.isfile(model_path):
        model = load_model(model_path,generator=data_generator)
    else:

        if model_type == "LEAP":
            ds_fac = 1
        else:
            ds_fac = 2

        train_generator = TrainingGenerator(generator=data_generator,
                                        downsample_factor=ds_fac,
                                        augmenter=None,
                                        sigma=5,
                                        validation_split=valid_size, 
                                        use_graph=True,
                                        random_seed=1,
                                        graph_scale=1)
        print("Training Generator loaded")

        if model_type == "CutResnet":
            model = DeepLabCut(train_generator, backbone="resnet50")
        elif model_type == "CutMobilenet":
            model = DeepLabCut(train_generator, backbone="mobilenetv2", alpha=1.0) # Increase alpha to improve accuracy
        elif model_type == "CutDensenet":
            model = DeepLabCut(train_generator, backbone="densenet121")
        elif model_type == "StackedDensenet":
            model = StackedDenseNet(train_generator, n_stacks=1, growth_rate=48)
        elif model_type == "LEAP":
            model = LEAP(train_generator)
        elif model_type == "StackedHourglass":
            model = StackedHourglass(train_generator)

    print("Model Set")

    logger = Logger(validation_batch_size=batch_size,
        # filepath saves the logger data to a .h5 file
        filepath=p().LOG
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=7)

    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        verbose=1,
        save_best_only=True,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        # monitor="loss" # use if validation_split=0
        min_delta=0.001,
        patience=100,
        verbose=1
    )

    callbacks = [logger, early_stop, reduce_lr, model_checkpoint]
    print("Callbacks set")


    import webbrowser
    webbrowser.open('https://www.youtube.com/watch?v=IkdmOVejUlI')

    model.fit(
        batch_size=batch_size,
        validation_batch_size=batch_size,
        callbacks=callbacks,
        epochs=1000,
        n_workers=workerCount(),
        steps_per_epoch=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('set', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    parser.add_argument('--model', type=str, choices=["CutResnet","CutMobilenet","CutDensenet","StackedDensenet","LEAP","StackedHourglass"],
                        default='StackedDensenet', help="The type of model to train."
                        )
    parser.add_argument('--batch',type=int, choices=[1,2,4,8,12,16], default=2, help="Batch size for training")
    parser.add_argument('--valid',type=float, default=0.2, help="Validation size for training")
    args = parser.parse_args()

    run(args.set, args.skeleton, args.model, args.batch, args.valid)