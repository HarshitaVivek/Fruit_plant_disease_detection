// Initialize/Load model
async function initialize() {
    let status = document.querySelector('.init_status')
    status.innerHTML = 'Loading Model .... <span class="fa fa-spinner fa-spin"></span>'
    // Load the Tensorflow.js model from the model.json file
    model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
    status.innerHTML = 'Model Loaded Successfully  <span class="fa fa-check"></span>'
}
// Loading the class_indices.json file which contains the Class names and encoded index
async function fetchData(){
    let response = await fetch('./class_indices.json');
    let data = await response.json();
    data = JSON.stringify(data);
    data = JSON.parse(data);
    return data;
}

// Predict function
async function predict() {
    // Function for invoking prediction
    let img = document.getElementById('image')
    // Scalar (0-rank tensor for scaling the image for uniformity with the images trained on)
    let offset = tf.scalar(255)
    // Load the uploaded image, then
    // Resize to the required shape (224,224), then
    // Convert data-type to Float and Expand dimensions
    let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().expandDims();
    // Re-scaling the image (Dividing by 255.0)
    let tensorImg_scaled = tensorImg.div(offset)
    prediction = await model.predict(tensorImg_scaled).data();

    fetchData().then((data)=> 
        {
            // Generate the argmax from the predictions tensor
            predicted_class = tf.argMax(prediction)

            class_idx = Array.from(predicted_class.dataSync())[0]
            document.querySelector('.pred_class').innerHTML = data[class_idx]
            document.querySelector('.inner').innerHTML = `${parseFloat(prediction[class_idx]*100).toFixed(2)}% SURE`
            console.log(data[class_idx])
            console.log(prediction[class_idx])

            progressBar.animate(prediction[class_idx]); // Indicator for the accuracy

        }
    );
          
      }