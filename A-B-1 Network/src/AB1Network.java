import java.lang.Math;
/**
 * The AB1Network class is a simple artificial neural network with a single hidden layer,
 * capable of learning basic logical functions (AND, OR, XOR) through training on a provided dataset.
 * It utilizes a sigmoid activation function and gradient descent for weight optimization.
 *
 * @author Kai Hong
 *
 * Date of Creation: 01/26/2024
 *
 */
public class AB1Network
{
   private int inputNodes;                       // Number of nodes in the input layer
   private int hiddenNodes;                      // Number of nodes in the hidden layer
   private int numTestCases;                     // Number of test cases
   private double[] hiddenLayer;                 // Values stored in the hidden layer
   private int numHiddenLayers;                  // Number of hidden layers (not used in this lab)
   private int numConnectivityLayers;            // Number of connectivity layers
   private double weightRangeStart;              // Lower bound of the random weight range
   private double weightRangeEnd;                // Upper bound of the random weight range
   private int maxIterations;                    // Maximum number of training iterations
   private double errorThreshold;                // Threshold for error termination
   private double lambda;                        // Learning rate for weight updates
   private double[] inputs;                      // One column of inputs
   private double[][] thetas;                    // 2-D array for all theta values
   private double[] omegas;                      // Array for all omega values
   private double[] psis;                        // Array for all psi values
   private double[][] inputToHiddenWeights;      // Weights between input and hidden layer
   private double[][] deltaInputToHiddenWeights; // Delta of input to hidden weights
   private double[] hiddenToOutputWeights;       // Weights between hidden and output layer
   private double[] deltaHiddenToOutputWeights;  // Delta of hidden to output weights
   private boolean isTraining;                   // Indicates if the network is in training mode
   private double[][] inputDataSet;              // Input dataset
   private double[] targetOutput;                // Target output dataset
   private double averageError;                  // Average error for the network
   private int numIterations;                    // Tracks the number of iterations
   private double[] outputValues;                // Outputs of the network
   private boolean maxIterationsReached;         // Indicates if max iterations limit was reached
   private boolean randLoadWeights;              // Indicates if the weights are loaded randomly or manually

   /**
    * Configures the neural network with specified parameters.
    */
   public void setNetworkConfig()
   {
      this.randLoadWeights = true;
      this.isTraining = true;
      this.inputNodes = 2;
      this.hiddenNodes = 1;
      this.numTestCases = 4;
      this.numConnectivityLayers = 2;
      this.numHiddenLayers = 1;
      this.weightRangeStart = -1.5;
      this.weightRangeEnd = 1.5;
      this.maxIterations = 100000;
      this.errorThreshold = 0.0002;
      this.lambda = 0.3;
      averageError = Integer.MAX_VALUE;
   } // public void setNetworkConfig()


   /**
    * Prints the current configuration of the neural network to the console.
    *
    * @param isTraining Indicates whether the network is currently in training mode.
    */
   public void printNetworkConfig(boolean isTraining)
   {
      System.out.println("Network Configuration: " + inputNodes + "-" + hiddenNodes + "-1");
      if (isTraining)
      {
         System.out.println("Weight range: " + weightRangeStart + " - " + weightRangeEnd);
         System.out.println("Max iterations: " + maxIterations);
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Lambda value: " + lambda);
      } // if(isTraining)
      System.out.println();
   } // public void printNetworkConfig(boolean isTraining)


   /**
    * Allocates memory for and initializes the instance variables, datasets, weight matrices, and
    * other arrays required for the network's operation.
    */
   public void memoryAllocation()
   {
      inputDataSet = new double[numTestCases][inputNodes];
      inputs = new double[numTestCases];
      targetOutput = new double[numTestCases];

      inputToHiddenWeights = new double[inputNodes][hiddenNodes];
      hiddenToOutputWeights = new double[hiddenNodes];
      outputValues = new double[numTestCases];
      hiddenLayer = new double[hiddenNodes];

      if (isTraining)
      {
         int maxNum = Math.max(inputNodes, hiddenNodes);
         thetas = new double[numConnectivityLayers][hiddenNodes];
         omegas = new double[hiddenNodes];
         psis = new double[hiddenNodes];
         deltaInputToHiddenWeights = new double[maxNum][maxNum];
         deltaHiddenToOutputWeights = new double[maxNum];
      } //if (isTraining)
   } // public void memoryAllocation()


   /**
    * Populates all arrays of the network (input dataset, input-to-hidden and hidden-to-output weights)
    * with random or predetermined values within the specified range.
    */
   public void populateArrays()
   {
      inputDataSet[0][0] = 0;
      inputDataSet[0][1] = 0;

      inputDataSet[1][0] = 0;
      inputDataSet[1][1] = 1;

      inputDataSet[2][0] = 1;
      inputDataSet[2][1] = 0;
      
      inputDataSet[3][0] = 1;
      inputDataSet[3][1] = 1;

      /*
       * Currently configured for OR
       */
      targetOutput[0] = 0.0;
      targetOutput[1] = 1.0;
      targetOutput[2] = 1.0;
      targetOutput[3] = 0.0;

      if (randLoadWeights)
      {
         for (int k = 0; k < inputNodes; k++)
         {
            for (int j = 0; j < hiddenNodes; j++)
            {
               inputToHiddenWeights[k][j] = getRandomWeightInRange();
            }
         } // if(randLoadWeights)

         for (int k = 0; k < hiddenNodes; k++)
         {
            hiddenToOutputWeights[k] = getRandomWeightInRange();
         } // for (int k = 0; k < hiddenNodes; k++)
      }
      else
      {
         inputToHiddenWeights[0][0] = 0.1;
         inputToHiddenWeights[0][1] = 0.2;
         inputToHiddenWeights[1][0] = 0.3;
         inputToHiddenWeights[1][1] = 0.4;
         hiddenToOutputWeights[0] = 0.5;
         hiddenToOutputWeights[1] = 0.6;
      } // if(randLoadWeights)
   } // public void populateArrays()


   /**
    * Generates a random weight value within the specified range set during network configuration.
    *
    * @return A randomly generated weight value.
    */
   private double getRandomWeightInRange()
   {
      return Math.random() * (weightRangeEnd - weightRangeStart) + weightRangeStart;
   }


   /**
    * Runs the network operation by forward passing with all inputs and applying randomized or predetermined weights.
    */
   public void runNetwork()
   {
      System.out.println("Running the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");
      for (int runIndex = 0; runIndex < numTestCases; runIndex++)
      {
         double theta = 0.0;

         for (int k = 0; k < inputNodes; k++)
         {
            inputs[k] = inputDataSet[runIndex][k];
         } // for (int k = 0; k < inputNodes; k++)

         for (int j = 0; j < hiddenNodes; j++)
         {
            for (int k = 0; k < inputNodes; k++)
            {
               theta += inputs[k] * inputToHiddenWeights[k][j];
            }
         } // for (int j = 0; j < hiddenNodes; j++)

         for (int j = 0; j < hiddenNodes; j++)
         {
            hiddenLayer[j] = sigmoid(theta);
         }

         theta = 0.0; // zeroing the accumulator

         for (int j = 0; j < hiddenNodes; j++)
         {
            theta += hiddenLayer[j] * hiddenToOutputWeights[j];
         } // for (int j = 0; j < hiddenNodes; j++)
         outputValues[runIndex] = sigmoid(theta);

      } // for (int runIndex = 0; runIndex < numTestCases; runIndex++)
      System.out.println();
      System.out.println("Running Completed.");
   } // public static void runModel()


   /**
    * Starts the training process of the neural network using the dataset, target outputs, and training parameters previously set.
    * It performs forward passes, computes errors, saves the errors in errors, and
    * updates weights using gradient descent until the specified conditions are met.
    */
   public void trainNetwork()
   {
      System.out.println("Training the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");
      System.out.println();

      while ((numIterations < maxIterations) && (averageError > errorThreshold))
      {
         averageError = 0.0;
         for (int trainIndex = 0; trainIndex < numTestCases; trainIndex++)
         {
            for (int k = 0; k < inputNodes; k++)
            {
               inputs[k] = inputDataSet[trainIndex][k];
            } //for (int k = 0; k < inputNodes; k++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               thetas[0][j] = 0.0;
               for (int k = 0; k < inputNodes; k++)
               {
                  thetas[0][j] += inputs[k] * inputToHiddenWeights[k][j];
               }
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               hiddenLayer[j] = sigmoid(thetas[0][j]);
            } // for (int j = 0; j < hiddenNodes; j++)

            thetas[1][0] = 0.0;  // zeroing the accumulator

            for (int j = 0; j < hiddenNodes; j++)
            {
               thetas[1][0] += hiddenLayer[j] * hiddenToOutputWeights[j];

            } // for (int j = 0; j < hiddenNodes; j++)

            outputValues[trainIndex] = sigmoid(thetas[1][0]);

            averageError += errorFunction(outputValues[trainIndex], targetOutput[trainIndex]);

            double smallOmega = targetOutput[trainIndex] - outputValues[trainIndex];
            double smallPsi = smallOmega * sigmoidDerivative(thetas[1][0]);

            for (int j = 0; j < hiddenNodes; j++)
            {
               omegas[j] = smallPsi * hiddenToOutputWeights[j];

            } // for (int j = 0; j < hiddenNodes; j++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               psis[j] = omegas[j] * sigmoidDerivative(thetas[0][j]);

            } // for (int j = 0; j < hiddenNodes; j++)


            for (int j = 0; j < hiddenNodes; j++)
            {
               double changeInError = -hiddenLayer[j] * smallPsi;
               deltaHiddenToOutputWeights[j] = -lambda * changeInError;
            } // for (int j = 0; j < hiddenNodes; j++)


            for (int k = 0; k < inputNodes; k++)
            {
               for (int j = 0; j < hiddenNodes; j++)
               {
                  double changeInError = -inputs[k] * psis[j];
                  deltaInputToHiddenWeights[k][j] = -lambda * changeInError;
               } // for (int k = 0; k < inputNodes; k++)
            }

            for (int k = 0; k < inputNodes; k++)
            {
               for (int j = 0; j < hiddenNodes; j++)
               {
                  inputToHiddenWeights[k][j] += deltaInputToHiddenWeights[k][j];
               } // for (int j = 0; j < hiddenNodes; j++)
            } // for(int k = 0; k < inputNodes; k++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               hiddenToOutputWeights[j] += deltaHiddenToOutputWeights[j];
            } // for (int j = 0; j < hiddenNodes; j++)

         } // for (int i = 0; i < inputSetSize; i++)
         averageError /= (double) numTestCases;
         numIterations++;
      } // while ((numIterations < maxIterations) && (averageError > errorThreshold))

      maxIterationsReached = numIterations >= maxIterations;
      System.out.println("Training completed.");
      System.out.println();
   } // public void trainNetwork()


   /**
    * Computes the error for a single training example using the squared error function.
    *
    * @param expected The expected output value for the training example.
    * @param actual The actual output value produced by the network.
    * @return The computed error for the training example.
    */
   public double errorFunction(double expected, double actual)
   {
      double error = expected - actual;
      return 0.5 * error * error;
   }


   /**
    * The sigmoid activation function, used to calculate the output of neurons.
    *
    * @param input The input value to the neuron.
    * @return The output of the neuron after applying the sigmoid function.
    */
   public double sigmoid(double input)
   {
      return 1.0 / (1.0 + Math.exp(-input));
   }


   /**
    * Computes the derivative of the sigmoid function, necessary for backpropagation.
    *
    * @param output The output of the neuron.
    * @return The derivative of the sigmoid function at the given output value.
    */
   public double sigmoidDerivative(double output)
   {
      double sig = sigmoid(output);
      return sig * (1.0 - sig);
   }


   /**
    * Prints the results of the training process or running process.
    * If training, results include the final error, number of iterations,
    * and a truth table showing the network's outputs.
    */
   public void reportResults()
   {
      if (isTraining)
      {
         System.out.print("Termination reason: ");

         if (maxIterationsReached)
         {
            System.out.println("Maximum number of iterations (" + maxIterations + ") reached.");
         }
         else
         {
            System.out.println("Error threshold (" + errorThreshold + ") reached.");
         } // if (maxIterationsReached)
         System.out.println("Number of iterations: " + numIterations);

      } // if (isTraining)

      System.out.println();
      System.out.println("Average error: " + averageError);
      System.out.println();

      if(isTraining)
      {
         System.out.println("Training Results");
      }
      else
      {
         System.out.println("Running Results");
      } // if(isTraining)

      System.out.println("----------------");

      for (int index = 0; index < numTestCases; index++)
      {
         System.out.println("Test Case " + (index + 1) + ":");
         for (int colNum = 0; colNum < inputNodes; colNum++)
         {
            System.out.println("I" + (colNum + 1) + ": " + inputDataSet[index][colNum]);
         }
         System.out.println("Expected Value: " + targetOutput[index]);
         System.out.println("Actual Value: " + outputValues[index]);
         System.out.println();
      } // for (int index = 0; index < numTestCases; index++)

      System.out.println();
   } // public void printTrainingResults()


   /**
    * The main method for the AB1Network class.
    * It creates an instance of the network, configures it,
    * trains it, and prints the results.
    * @param args Command line arguments
    */
   public static void main(String[] args)
   {
      AB1Network network = new AB1Network();
      network.setNetworkConfig();
      network.memoryAllocation();
      network.populateArrays();
      network.printNetworkConfig(network.isTraining);

      if (network.isTraining)
      {
         network.trainNetwork();
      }
      else
      {
         network.runNetwork();
      } // if (network.isTraining)
      network.reportResults();
   } // public static void main(String args[])
} // public class AB1Network