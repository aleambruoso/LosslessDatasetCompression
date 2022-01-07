<h2>Lossless Dataset Compression</h2><br>
<br>
<p>Development of the lossless dataset compression method proposed by Borowsky, Marion and P. Calmon. <br>
We developed the lossless dataset compression method proposed by Borowsky, Marion and P. Calmon and tested on the MNIST dataset.<br>
The python script is executable from the command line after installing the dependencies.<br>
Below is the composition of the root of the project:</p><br>
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