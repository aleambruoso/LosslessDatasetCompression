<h2>Lossless Dataset Compression</h2><br>
<br>
<p>Sviluppo del metodo di compressione lossless dataset proposto da Borowsky, Marion e P. Calmon. <br>
Abbiamo sviluppato il metodo di compressione lossless dataset proposto da Borowsky, Marion e P. Calmon e testato sul dataset MNIST.<br>
Lo script python è eseguibile da linea comando previa installazione della dipendenze.<br>
Di seguito la composizione della root del progetto:</p><br>
<br>
root:/
|_code: DatasetCompressor.py
|_MNIST model: model/
| |_checkpoint
| |_gru_128.data-00000-of-00001
| |_gru_128.index
| |_gru_128.meta
|_MNIST dataset: mnist/
| |_train: trainFile.npy
| |_test: testFile.npy
|_MNIST compressed: mnistEncoded.npy
|_documentation: documentation/
| |_Paper: Predictive_Coding_for_Lossless_Dataset_Compression.pdf
| |_Relation: Lossless_Dataset_Compression.pdf