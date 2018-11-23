const MODEL_PATH = './tfjsv3/model.json';
const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let model;
const modelDemo = async () => {
  status('Loading model...');
  model = await tf.loadModel(MODEL_PATH);
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  status('Loaded');
};

/**
 * Given an image element, makes a prediction through model returning the
 * probabilities of the top K classes.
 */
async function predict(imgElementx) {
  status('Predicting...');

  const startTime = performance.now();
  const logits = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    const img = tf.fromPixels(imgElementx).toFloat();
    img.print;
    const offset = tf.scalar(255.0);
    const su = tf.tensor([103.939 / 255, 116.779 / 255, 123.68 / 255], shape=([1, 1, 3]), dtype=tf.float32);
    su.print();
    // const normalized = img.sub(offset).div(offset);
    const normalized = img.div(offset).sub(su);
    //normalized.print();
    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    // console.log(batched);
    
    // Make a prediction through model.
    return model.predict(batched);
  });
  const values = await logits.data();
  console.log(values);
  const totalTime = performance.now() - startTime;
  status(`Done in ${Math.floor(totalTime)}ms`);
  predictionsElement.innerText = values[0]*100;
  // Show the classes in the DOM.
}

const predictImg = (img) => {
  // inception model works with 224 x 224 images, so we resize
  // our input images and pad the image with white pixels to
  // make the images have the same width and height
  const maxImgDim = 224;
  const white = new cv.Vec(255, 255, 255);
  const imgResized = img.resizeToMax(maxImgDim).padToSquare(white);

  // network accepts blobs as input
  const inputBlob = cv.blobFromImage(imgResized);
  net.setInput(inputBlob);
  // forward pass input through entire network, will return
  // classification result as 1xN Mat with confidences of each class
  const outputBlob = net.forward();
  // find all labels with a minimum confidence
  const minConfidence = 0.05;
  const locations =
    outputBlob
      .threshold(minConfidence, 1, cv.THRESH_BINARY)
      .convertTo(cv.CV_8U)
      .findNonZero();

  const result =
    locations.map(pt => ({
      confidence: parseInt(outputBlob.at(0, pt.x) * 100) / 100,
      className: classNames[pt.x]
    }))
      // sort result by confidence
      .sort((r0, r1) => r1.confidence - r0.confidence)
      .map(res => `${res.className} (${res.confidence})`);

  return result;
}


//
// UI
//

var imgElement = document.getElementById('imageSrc');
var img2Element = document.getElementById('canvasOutput');
var inputElement = document.getElementById('fileInput');

const filesElement = document.getElementById('fileInput');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  imgElement.src = URL.createObjectURL(evt.target.files[0]);
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
      
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

modelDemo();