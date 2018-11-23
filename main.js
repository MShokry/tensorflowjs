const MODEL_PATH = './tfjsv2/model.json';
const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;
/* Loading the model */
let model;
const modelDemo = async () => {
  status('Loading model...');
  model = await tf.loadModel(MODEL_PATH);
  // model.summary();
  //for warmingup the model no output
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  status('Loaded');
};
/**
 * Given an image element, makes a prediction through model returning 0:100
 */
async function predict(imgElementx) {
  status('Predicting...');
  const startTime = performance.now();
  const logits = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    const img = tf.fromPixels(imgElementx).toFloat();
    const img1 = tf.reverse(img, axis = [-1])
    // img1.print();
    const offset = tf.scalar(255.0);
    const su = tf.tensor([103.939 / 255, 116.779 / 255, 123.68 / 255], shape = ([1, 1, 3]), dtype = 'float32');
    // su.print();
    const normalized = img1.div(offset).sub(su);
    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    // Make a prediction through model.
    return model.predict(batched);
  });
  const values = await logits.data();
  // console.log(values);
  const totalTime = performance.now() - startTime;
  // Show the prediction in the DOM.
  status(`Done in ${Math.floor(totalTime)}ms`);
  predictionsElement.innerText = values[0] * 100;
  return values[0] * 100;
}
//
// UI
//
var imgElement = document.getElementById('imageSrc');
var img2Element = document.getElementById('canvasOutput');
var inputElement = document.getElementById('fileInput');

inputElement.addEventListener('change', (e) => {
  if (inputElement.files.length === 0) {
    output.value = 'No file selected';
    return;
  }
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

let mat;
imgElement.onload = function () {
  
  let img2 = cv.imread(imgElement);
  let dsize = new cv.Size(IMAGE_SIZE, IMAGE_SIZE);
  let dest = new cv.Mat();
  cv.resize(img2, dest, dsize, 0, 0);
  cv.imshow(img2Element, dest);
  let imgData = new ImageData(new Uint8ClampedArray(dest.data), IMAGE_SIZE, IMAGE_SIZE);
  // console.log(imgData);
  const pr = predict(imgData);
  // Use the prediction for any operations ###
  //
};

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');


$(document).ready(function () {
  modelDemo();
});