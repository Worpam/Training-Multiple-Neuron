

//generate input numbers from 1 to 20;
const INPUTS = [];
for (let n =1; n <= 20; n++) {
	INPUTS.push(n);
}
//generate outpusts that are simply an input multiplied by itself
//to generate some non linear data.
const OUTPUTS = [];
for (let n = 0; n < INPUTS.length; n++) {
	OUTPUTS.push(INPUTS[n] * INPUTS[n]);
}

//input feature array is 1 dimensional
const INPUTS_TENSOR=tf.tensor1d(INPUTS);
const OUTPUTS_TENSOR=tf.tensor1d(OUTPUTS);

function normalize(tensor, min, max){
    const result=tf.tidy(function(){
        const MIN_VALUES=min || tf.min(tensor,0);
        const MAX_VALUES=max || tf.max(tensor,0);
        const TENSOR_SUB_MIN_VALUES=tf.sub(tensor, MIN_VALUES);
        const RANGE=tf.sub(MAX_VALUES, MIN_VALUES);
        const NORMALIZE_VALUE=tf.div(TENSOR_SUB_MIN_VALUES, RANGE);
        
        return {NORMALIZE_VALUE, MIN_VALUES, MAX_VALUES};


    });
    return result;
}

const FEATURE_RESULTS= normalize(INPUTS);

console.log("Normalized Values: ");
FEATURE_RESULTS.NORMALIZE_VALUE.print();

console.log("Minimum Values: ");
FEATURE_RESULTS.MIN_VALUES.print();

console.log("Maximum values: ");
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();

const model=tf.sequential();

//we will use one dense layer with 3 neuron (units) and an input of
//1input feature to match the input values
model.add(tf.layers.dense({inputShape: [1], units:200, activation: 'relu'}));//activatio 'relu' for all neurons in this function

//Adding a new hidden layer with 100neurons and relu activation.
model.add(tf.layers.dense({units:100, activation:'relu'}));
//add another dense layer with 1 neuron that will be connected
//to the first input layer above
model.add(tf.layers.dense({units:1}))//this is connected to the hidden 100 neurons
model.summary();


//global variable
const LEARNING_RATE=0.0001;//0.01 will return nan
const OPTIMIZER=tf.train.sgd(LEARNING_RATE);
train();

async function train(){
    

    model.compile({
        optimizer:OPTIMIZER,
        loss:"meanSquaredError",
    });

    let result=await model.fit(FEATURE_RESULTS.NORMALIZE_VALUE,OUTPUTS_TENSOR, {
        shuffle:true,
        batchSize:2,
        epochs:200, //go over the data 200 times!
        callbacks:{onEpochEnd: logProgress}//to use for key events(specify a function to call everytime an epoch ends) --see line 78


    });
    
    OUTPUTS_TENSOR.dispose();
    FEATURE_RESULTS.NORMALIZE_VALUE.dispose();

    console.log("Final Average error loss: " + Math.sqrt(result.history.loss[result.history.loss.length - 1]));

    evaluate();
    
}



function evaluate(){
    tf.tidy(function(){
        let new_inputs=normalize(tf.tensor1d([7]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);
        let output=model.predict(new_inputs.NORMALIZE_VALUE);
        output.print();
    });
    FEATURE_RESULTS.MIN_VALUES.dispose();
    FEATURE_RESULTS.MAX_VALUES.dispose();
    model.dispose();
    

}
//logProgress function
function logProgress(epoch, logs){//current epoch no and logs of training it has
    console.log("Data for epoch: " + epoch, Math.sqrt(logs.loss));

    //setting a new learning rate when epoch reach 70
    if(epoch==70){
        OPTIMIZER.setLearningRate(LEARNING_RATE/2);//allows us to go down lower
    }
 }
//trying to predict 7*7 = 49



