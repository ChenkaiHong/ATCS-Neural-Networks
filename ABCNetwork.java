import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.Math;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * The ABCNetwork class is a three-layer artificial neural network with a single hidden layer.
 * It is capable of learning logical functions (such as AND, OR, XOR) through training on a provided dataset
 * and uses a sigmoid activation function and gradient descent for weight optimization.
 *
 * Revision 1:
 *    1. Zeroing is now completed before the loops that use them in runNetwork() and trainNetwork()
 *    2. Removed the initialization of averageError in setNetworkConfig()
 *    3. Optimized flow control constructs by adding blank lines in necessary locations
 *    4. Made maxIterationsReached and errorThresholdReached independent of each other
 *    5. Optimized the printing of training results by implementing a table
 *    6. Implemented a weight-saving procedure that saves trained weights in a .txt file
 *
 * @author Kai Hong
 *
 * Date of Creation: 03/01/2024
 *
 */
public class ABCNetwork
{
   private int inputNodes;                        // Number of nodes in the input layer
   private int hiddenNodes;                       // Number of nodes in the hidden layer
   private int outputNodes;                       // Number of nodes in the output layer

   private int numTestCases;                      // Number of test cases

   private double[] hiddenLayer;                  // Values stored in the hidden layer
   private int numHiddenLayers;                   // Number of hidden layers (not used in this lab)
   private int numConnectivityLayers;             // Number of connectivity layers

   private double weightRangeStart;               // Lower bound of the random weight range
   private double weightRangeEnd;                 // Upper bound of the random weight range

   private int maxIterations;                     // Maximum number of training iterations
   private double errorThreshold;                 // Threshold for error termination
   private double lambda;                         // Learning rate for weight updates

   private double[] inputs;                       // One column of inputs
   private double[][] trainThetas;                // 2-D array for all training theta values
   private double[][] runThetas;                  // 2-D array for all running theta values
   private double[] omegas;                       // Array for all omega values
   private double[] smallOmegas;                  // Array for all small omegas values
   private double[] psis;                         // Array for all psi values
   private double[] smallPsis;                    // Array for all small psi values

   private double[][] inputToHiddenWeights;       // Weights between input and hidden layer
   private double[][] deltaInputToHiddenWeights;  // Delta of input to hidden weights
   private double[][] hiddenToOutputWeights;      // Weights between hidden and output layer
   private double[][] deltaHiddenToOutputWeights; // Delta of hidden to output weights

   private boolean isTraining;                    // Indicates if the network is in training mode
   private boolean contTraining;                  // Indicates if the network should keep training

   private double[][] inputDataSet;               // Input dataset
   private double[][] targetOutput;               // Target output dataset
   private double[][] outputValues;               // Outputs of the network

   private double averageError;                   // Average error for the network
   private int numIterations;                     // Tracks the number of iterations

   private boolean errorThresholdReached;         // Indicates if the error threshold was reached
   private boolean maxIterationsReached;          // Indicates if max iterations limit was reached

   private File savedWeights;                     // Text file where all weights are saved to


   /**
    * Configures the neural network with specified parameters.
    */
   public void setNetworkConfig()
   {
      this.isTraining = true;

      this.inputNodes = 3;
      this.hiddenNodes = 1;
      this.outputNodes = 3;

      this.numTestCases = 4;
      this.numConnectivityLayers = 2;
      this.numHiddenLayers = 1;

      this.weightRangeStart = 0.1;
      this.weightRangeEnd = 1.5;

      this.maxIterations = 100000;
      this.errorThreshold = 0.0002;

      this.lambda = 0.3;

      savedWeights = new File("/Users/kh/Desktop/Shared/ABCNetwork/src/savedWeights.txt");
   } // public void setNetworkConfig()


   /**
    * Prints the current configuration of the neural network to the console.
    *
    * @param isTraining Indicates whether the network is currently in training mode.
    */
   public void printNetworkConfig(boolean isTraining)
   {
      System.out.println("Network Configuration: " + inputNodes + "-" + hiddenNodes + "-" + outputNodes);

      if (isTraining)
      {
         System.out.println("Weight range: " + weightRangeStart + " - " + weightRangeEnd);
         System.out.println("Max iterations: " + maxIterations);
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Lambda value: " + lambda);
      } // if (isTraining)

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
      targetOutput = new double[numTestCases][outputNodes];
      runThetas = new double[numConnectivityLayers][hiddenNodes];

      inputToHiddenWeights = new double[inputNodes][hiddenNodes];
      hiddenToOutputWeights = new double[hiddenNodes][outputNodes];
      outputValues = new double[numTestCases][outputNodes];
      hiddenLayer = new double[hiddenNodes];

      if (isTraining)
      {
         contTraining = true;

         int maxNum = Math.max(inputNodes, hiddenNodes);
         maxNum = Math.max(maxNum, outputNodes);

         trainThetas = new double[numConnectivityLayers][maxNum];
         omegas = new double[hiddenNodes];
         smallOmegas = new double[outputNodes];
         psis = new double[hiddenNodes];
         smallPsis = new double[outputNodes];

         deltaInputToHiddenWeights = new double[maxNum][maxNum];
         deltaHiddenToOutputWeights = new double[maxNum][maxNum];
      } //if (isTraining)
   } // public void memoryAllocation()


   /**
    * Populates all arrays of the network (input dataset, input-to-hidden and hidden-to-output weights)
    * with random values within the specified range.
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

      numIterations = 0;

      /*
       * Currently configured for AND, OR, and XOR
       */
      targetOutput = new double[][] {
              {0, 0, 0},
              {0, 1, 1},
              {0, 1, 1},
              {1, 1, 0}
      };

      for (int k = 0; k < inputNodes; k++)
      {
         for (int j = 0; j < hiddenNodes; j++)
         {
            inputToHiddenWeights[k][j] = getRandomWeightInRange();
         }
      } // for (int k = 0; k < inputNodes; k++)

      for (int k = 0; k < hiddenNodes; k++)
      {
         for (int i = 0; i < outputNodes; i++)
         {
            hiddenToOutputWeights[k][i] = getRandomWeightInRange();
         }
      } // for (int k = 0; k < hiddenNodes; k++)
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


   /*
    * Saves the weights in a file named savedWeights.txt
    */
   public void saveWeights() throws FileNotFoundException, UnsupportedEncodingException
   {
      try (PrintWriter writer = new PrintWriter(savedWeights))
      {
         for (int k = 0; k < inputNodes; k++)
         {
            for (int j = 0; j < hiddenNodes; j++)
            {
               writer.println(inputToHiddenWeights[k][j]);
            }
         } // for (int k = 0; k < inputNodes; k++)

         for (int j = 0; j < hiddenNodes; j++)
         {
            for (int i = 0; i < outputNodes; i++)
            {
               writer.println(hiddenToOutputWeights[j][i]);
            }
         } // for (int j = 0; j < hiddenNodes; j++)
      } //try (PrintWriter writer = new PrintWriter(savedWeights))
   } //public void saveWeights() throws FileNotFoundException, UnsupportedEncodingException


   /*
    * Loads the weights from savedWeights.txt
    */
   public void loadWeights() throws FileNotFoundException
   {
      Scanner sc = new Scanner(savedWeights);

      for (int k = 0; k < inputNodes; k++)
      {
         for (int j = 0; j < hiddenNodes; j++)
         {
            inputToHiddenWeights[k][j] = sc.nextDouble();
         }
      } // for (int k = 0; k < inputNodes; k++)

      for (int j = 0; j < hiddenNodes; j++)
      {
         for (int i = 0; i < outputNodes; i++)
         {
            hiddenToOutputWeights[j][i] = sc.nextDouble();
         }
      } // for (int j = 0; j < hiddenNodes; j++)

      sc.close();
   } //public void loadWeights() throws FileNotFoundException


   /**
    * Runs the network operation by forward passing all inputs and applying trained weights
    */
   public void runNetwork()
   {
      try
      {
         loadWeights();
      }
      catch (FileNotFoundException ignored)
      {
      }

      System.out.println("Running the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");

      averageError = 0;

      for (int runIndex = 0; runIndex < numTestCases; runIndex++)
      {
         for (int k = 0; k < inputNodes; k++)
         {
            inputs[k] = inputDataSet[runIndex][k];
         } //for (int k = 0; k < inputNodes; k++)

         for (int j = 0; j < hiddenNodes; j++)
         {
            runThetas[0][j] = 0.0;  // zeroing the accumulator
         }

         for (int j = 0; j < hiddenNodes; j++)
         {
            for (int k = 0; k < inputNodes; k++)
            {
               runThetas[0][j] += inputs[k] * inputToHiddenWeights[k][j];
            }
         } // for (int j = 0; j < hiddenNodes; j++)

         for (int j = 0; j < hiddenNodes; j++)
         {
            hiddenLayer[j] = sigmoid(runThetas[0][j]);
         } // for (int j = 0; j < hiddenNodes; j++)

         for (int i = 0; i < outputNodes; i++)
         {
            runThetas[1][i] = 0.0;  // zeroing the accumulator
         } // for (int i = 0; i < outputNodes; i++)

         for (int j = 0; j < hiddenNodes; j++)
         {
            for (int i = 0; i < outputNodes; i++)
            {
               runThetas[1][i] += hiddenLayer[j] * hiddenToOutputWeights[j][i];
            }
         } // for (int j = 0; j < hiddenNodes; j++)

         for (int i = 0; i < outputNodes; i++)
         {
            outputValues[runIndex][i] = sigmoid(runThetas[1][i]);
            averageError += errorFunction(targetOutput[runIndex][i] - outputValues[runIndex][i]);
         } // for (int i = 0; i < outputNodes; i++)
      } // for (int runIndex = 0; runIndex < numTestCases; runIndex++)

      averageError /= (double) numTestCases;

      System.out.println();
      System.out.println("Running Completed.");
   } // public static void runModel()


   /**
    * Starts the training process using the dataset, target outputs, and training parameters previously set
    * It performs forward passes, computes errors, saves the errors, and
    * updates weights using gradient descent until the specified conditions are met.
    */
   public void trainNetwork()
   {
      System.out.println("Training the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");
      System.out.println();

      while (contTraining)
      {
         averageError = 0;

         for (int trainIndex = 0; trainIndex < numTestCases; trainIndex++)
         {
            for (int k = 0; k < inputNodes; k++)
            {
               inputs[k] = inputDataSet[trainIndex][k];
            } //for (int k = 0; k < inputNodes; k++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               trainThetas[0][j] = 0.0; // zeroing the accumulator
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               for (int k = 0; k < inputNodes; k++)
               {
                  trainThetas[0][j] += inputs[k] * inputToHiddenWeights[k][j];
               }
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               hiddenLayer[j] = sigmoid(trainThetas[0][j]);
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int i = 0; i < outputNodes; i++)
            {
               trainThetas[1][i] = 0.0;  // zeroing the accumulator
            } // for (int i = 0; i < outputNodes; i++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               for (int i = 0; i < outputNodes; i++)
               {
                  trainThetas[1][i] += hiddenLayer[j] * hiddenToOutputWeights[j][i];
               }
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int i = 0; i < outputNodes; i++)
            {
               outputValues[trainIndex][i] = sigmoid(trainThetas[1][i]);
               smallOmegas[i] = targetOutput[trainIndex][i] - outputValues[trainIndex][i];
               averageError += errorFunction(smallOmegas[i]);
               smallPsis[i] = smallOmegas[i] * sigmoidDerivative(trainThetas[1][i]);
            } // for (int i = 0; i < outputNodes; i++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               omegas[j] = 0.0; // zeroing the accumulator
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               for (int i = 0; i < outputNodes; i++)
               {
                  omegas[j] += smallPsis[i] * hiddenToOutputWeights[j][i];
               }
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               psis[j] = omegas[j] * sigmoidDerivative(trainThetas[0][j]);
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               for (int i = 0; i < outputNodes; i++)
               {
                  deltaHiddenToOutputWeights[j][i] = -lambda * -hiddenLayer[j] * smallPsis[i];
               }
            } // for (int j = 0; j < hiddenNodes; j++)

            for (int k = 0; k < inputNodes; k++)
            {
               for (int j = 0; j < hiddenNodes; j++)
               {
                  deltaInputToHiddenWeights[k][j] = -lambda * -inputs[k] * psis[j];
               }
            } // for (int k = 0; k < inputNodes; k++)

            for (int k = 0; k < inputNodes; k++)
            {
               for (int j = 0; j < hiddenNodes; j++)
               {
                  inputToHiddenWeights[k][j] += deltaInputToHiddenWeights[k][j];
               }
            } // for (int k = 0; k < inputNodes; k++)

            for (int j = 0; j < hiddenNodes; j++)
            {
               for (int i = 0; i < outputNodes; i++)
               {
                  hiddenToOutputWeights[j][i] += deltaHiddenToOutputWeights[j][i];
               }
            } // for (int j = 0; j < hiddenNodes; j++)
         } // for (int i = 0; i < inputSetSize; i++)

         averageError /= (double) numTestCases;
         numIterations++;

         if (averageError <= errorThreshold)
         {
            errorThresholdReached = true;
            contTraining = false;
         } //if (avgErr <= errThresh)

         if (numIterations >= maxIterations)
         {
            maxIterationsReached = true;
            contTraining = false;
         } //if (countIterations >= maxIterations)
      } // if (contTraining)

      try
      {
         saveWeights();
      }
      catch (FileNotFoundException | UnsupportedEncodingException ignored)
      {
      }

      System.out.println("Training completed.");
      System.out.println();
   } // public void trainNetwork()


   /**
    * Computes the error for a single training example using the squared error function.
    *
    * @param error The error for the training example.
    * @return The computed error for the training example.
    */
   public double errorFunction(double error)
   {
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
    * If training, results include the error reached, iterations reached, and reason for an end of run.
    * A truth table is outputted for both training and running.
    */
   public void reportResults()
   {
      if (isTraining)
      {
         System.out.print("Termination reason: ");

         if (maxIterationsReached)
         {
            System.out.println("Maximum number of iterations (" + maxIterations + ") reached.");
         } // if (maxIterationsReached)

         if (errorThresholdReached)
         {
            System.out.println("Error threshold (" + errorThreshold + ") reached.");
         } // if (maxIterationsReached)

         System.out.println("Number of iterations: " + numIterations);
      } // if (isTraining)

      System.out.println("Average error: " + averageError);
      System.out.println();

      if (isTraining)
      {
         System.out.println("\t\t\tTraining Results");
      }
      else
      {
         System.out.println("\t\t\tRunning Results");
      } // if (isTraining)

      System.out.println("--------------------------------------------");

      for (int colNum = 0; colNum < inputNodes; colNum++)
      {
         System.out.print("I" + (colNum + 1) + " |");
      }
      System.out.println("\tAND\t   |\tOR\t   |\tXOR\t   |");

      for (int index = 0; index < numTestCases; index++)
      {
         for (int colNum = 0; colNum < inputNodes; colNum++)
         {
            System.out.print(inputDataSet[index][colNum] + "|");
         } // for (int colNum = 0; colNum < inputNodes; colNum++)

         for (int i = 0; i < outputNodes; i++)
         {
            DecimalFormat formatter = new DecimalFormat("0.000000000");
            System.out.print(formatter.format(outputValues[index][i] ) + "|");
         } // for (int i = 0; i < outputNodes; i++)

         System.out.println();
      } // for (int index = 0; index < numTestCases; index++)

      System.out.println();
   } // public void reportResults()


   /**
    * The main method for the ABCNetwork class.
    * It creates an instance of the network, configures it,
    * trains or runs it, and prints the results.
    * @param args Command line arguments
    */
   public static void main(String[] args)
   {
      ABCNetwork network = new ABCNetwork();

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
} // public class ABCNetwork