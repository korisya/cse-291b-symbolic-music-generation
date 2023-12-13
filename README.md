# Music Transformer
Tested on PyTorch 1.13.0

## About
This is a reproduction of the MusicTransformer (Huang et al., 2018) for PyTorch. We propose adding a deep convolutional network to improve the learning of musical ideas, chord progression, and musical structure.

## Generated Music:
| Primer                                                                                                                           | Music Transformer                                                                                                                | Music Transformer + CNN
| -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------
| <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/40466f29-95ce-44e6-a4e8-3e290b746b38"> | <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/9409c36b-2a2e-4728-9fe3-0f20fdac221f"> | <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/7d3dc6d4-95e2-4bb9-a14c-13ff1da4d8c6">
| <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/f2e8ebb4-01e1-435c-9b55-f00ac3490fc1"> | <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/34a6c377-5a35-4491-b682-2964b78188d1"> | <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/e55e012c-7098-40f0-adb8-16efc463bd4c">
| <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/9e01390b-15bf-4111-ac0a-1a29d7bf5861"> | <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/45eb0443-10fa-41a1-9573-85acd8a65f8d"> | <video src="https://github.com/korisya/cse-291b-symbolic-music-generation/assets/42691569/17084a9f-c9ff-4f22-8887-782a40e51288">

## How to run
1. Download the Maestro dataset (we used v2 but v1 should work as well). You can download the dataset [here](https://magenta.tensorflow.org/datasets/maestro). You only need the MIDI version if you're tight on space. 

2. Run `git submodule update --init --recursive` to get the MIDI pre-processor provided by jason9693 et al. (https://github.com/jason9693/midi-neural-processor), which is used to convert the MIDI file into discrete ordered message types for training and evaluating. 

3. Run `preprocess_midi.py -output_dir <path_to_save_output> <path_to_maestro_data>`, or run with `--help` for details. This will write pre-processed data into folder split into `train`, `val`, and `test` as per Maestro's recommendation.

4. To train a model, run `train.py`. Use `--help` to see the tweakable parameters. See the results section for details on model performance. 

5. After training models, you can evaluate them with `evaluate.py` and generate a MIDI piece with `generate.py`. To graph and compare results visually, use `graph_results.py`.

### Training
As an example to train a model using the parameters specified in results:

```
python train.py -output_dir cnn --rpr --cnn
```
You can additonally specify both a weight and print modulus that determine what epochs to save weights and what batches to print. The weights that achieved the best loss and the best accuracy (separate) are always stored in results, regardless of weight modulus input.

### Evaluation
You can evaluate a model using;
```
python evaluate.py -model_weights cnn/results/best_acc_weights.pickle --rpr --cnn
```

Your model's results may vary because a random sequence start position is chosen for each evaluation piece. This may be changed in the future.

### Generation
You can generate a piece with a trained model by using:
```
python generate.py -output_dir output -model_weights cnn/results/best_acc_weights.pickle --rpr --cnn
```

The default generation method is a sampled probability distribution with the softmaxed output as the weights.
