<h2>Lossless Dataset Compression</h2><br>
<br>
<p>Sviluppo del metodo di compressione lossless dataset proposto da Borowsky, Marion e P. Calmon. <br>
Abbiamo sviluppato il metodo di compressione lossless dataset proposto da Borowsky, Marion e P. Calmon e testato sul dataset MNIST.<br>
Lo script python è eseguibile da linea comando previa installazione della dipendenze.<br>
Di seguito la composizione della root del progetto:</p><br>
<br>
root:/<br>
|_code: DatasetCompressor.py<br>
|_MNIST model: model/<br>
| |_checkpoint<br>
| |_gru_128.data-00000-of-00001<br>
| |_gru_128.index<br>
| |_gru_128.meta<br>
|_MNIST dataset: mnist/<br>
| |_train: trainFile.npy<br>
| |_test: testFile.npy<br>
|_MNIST compressed: mnistEncoded.npy<br>
|_documentation: documentation/<br>
| |_Paper: Predictive_Coding_for_Lossless_Dataset_Compression.pdf<br>
| |_Relation: Lossless_Dataset_Compression.pdf<br>