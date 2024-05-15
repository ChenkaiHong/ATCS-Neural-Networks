import java.io.*;
import java.lang.Math;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * The NLayer class is an N-layer artificial neural network with N hidden layers.
 * It is capable of learning logical functions (such as AND, OR, XOR) through training on a provided dataset
 * and uses a sigmoid activation function, gradient descent, and back propagation for weight optimization.
 *
 * Methods:
 *    setNetworkConfig(String controlFilePath)
 *    printNetworkConfig(boolean isTraining)
 *    memoryAllocation()
 *    findMaxNodesInRange(int startLayer, int endLayer)
 *    populateArrays()
 *    saveWeights()
 *    loadWeights()
 *    loadInputActivations(int index)
 *    forwardPass(int trainIndex)
 *    trainNetwork()
 *    runningOneTestCase(int runIndex)
 *    runNetwork()
 *    errorFunction(double error)
 *    activationFunction(double input)
 *    activationFunctionDerivative(double output)
 *    sigmoid(double input)
 *    sigmoidDerivative(double output)
 *    reportResults()
 *
 * @author Kai Hong
 *
 * Date of Creation: 04/26/2024
 */
public class NLayer
{
   private static final String DEFAULT_CONTROL_FILE = "NControl.txt";         // Default control file if no argument is passed
   private static final String PATH = "/Users/kh/Desktop/Shared/NLayer/src/"; // Default path of the source folder
   private int[] layers;                                                      // Number of nodes in the output layer
   private int numLayers;                                                     // Number of activation layers

   private int numTestCases;                                                  // Number of test cases

   private final int INPUTLAYER = 0;                                          // Index for the input layer
   private final int HIDDENLAYER1 = 1;                                        // Index for the first hidden layer
   private int outputLayer;                                                   // Index for the output layer
   private double weightRangeStart;                                           // Lower bound of the random weight range
   private double weightRangeEnd;                                             // Upper bound of the random weight range

   private int maxHiddenNodes;                                                // Max number of hidden nodes
   private int maxHiddenOutputNodes;                                          // Max number of hidden and output nodes
   private int maxNodes;                                                      // Max number of input, hidden, and output nodes

   private int maxIterations;                                                 // Maximum number of training iterations
   private double errorThreshold;                                             // Threshold for error termination
   private double lambda;                                                     // Learning rate for weight updates

   private double[][] thetas;                                                 // 2-D array for all training theta values
   private double[][] activationLayers;                                       // 2-D array for all values in the activation layers
   private double[][] psis;                                                   // 2-D array for all big psi values

   private double[][][] weights;                                              // Weights between input and hidden layer
   private boolean isTraining;                                                // Indicates if the network is in training mode
   private boolean contTraining;                                              // Indicates if the network should keep training

   private double[][] inputDataTable;                                         // Input dataset
   private double[][] outputDataTable;                                        // Output dataset
   private double[][] outputs;                                                // Final outputs of the network
   private double averageError;                                               // Average error for the network
   private int numIterations;                                                 // Tracks the number of iterations

   private boolean errorThresholdReached;                                     // Indicates if the error threshold was reached
   private boolean maxIterationsReached;                                      // Indicates if max iterations limit was reached

   private long elapsed;                                                      // Time it took for network to run or train
   private String expectedOutputsPath;                                        // Absolute file path of the expected outputs file
   private String inputFilePath;                                              // Absolute file path of the inputs file

   private String weightsConfig;                                              // Configuration of weights
   private boolean saveWeights;                                               // Indicates if weights are saved
   private String savedWeightsPath;                                           // Absolute path of the weights file

   private int keepAlive;                                                     // Keep alive value of the network

   /**
    * Configures the neural network with specified parameters.
    */
   public void setNetworkConfig(String controlFilePath) throws IOException
   {
      Scanner sc = new Scanner(new File(PATH + controlFilePath));

      sc.nextLine();
      isTraining = sc.nextBoolean();
      sc.nextLine();
      sc.nextLine();

      numLayers = sc.nextInt();
      sc.nextLine();
      sc.nextLine();

      outputLayer = numLayers - 1;

      layers = new int[numLayers];

      for (int index = 0; index < numLayers; index++)
      {
         layers[index] = sc.nextInt();
         sc.nextLine();
      }

      sc.nextLine();
      numTestCases = sc.nextInt();
      sc.nextLine();
      sc.nextLine();

      weightRangeStart = sc.nextDouble();
      sc.nextLine();
      sc.nextLine();
      weightRangeEnd = sc.nextDouble();
      sc.nextLine();
      sc.nextLine();

      maxIterations = sc.nextInt();
      sc.nextLine();
      sc.nextLine();
      errorThreshold = sc.nextDouble();
      sc.nextLine();
      sc.nextLine();
      lambda = sc.nextDouble();
      sc.nextLine();
      sc.nextLine();

      expectedOutputsPath = sc.next();
      sc.nextLine();
      sc.nextLine();
      inputFilePath = sc.next();
      sc.nextLine();
      sc.nextLine();

      weightsConfig = sc.next();
      sc.nextLine();
      sc.nextLine();
      saveWeights = sc.nextBoolean();
      sc.nextLine();
      sc.nextLine();
      savedWeightsPath = sc.next();
      sc.nextLine();
      sc.nextLine();

      keepAlive = sc.nextInt();
   } // public void setNetworkConfig(String controlFilePath) throws IOException

   /**
    * Prints the current configuration of the neural network to the console.
    *
    * @param isTraining Indicates whether the network is currently in training mode.
    */
   public void printNetworkConfig(boolean isTraining)
   {
      System.out.println("Network Configuration: ");
      System.out.print(layers[INPUTLAYER]);

      for (int index = HIDDENLAYER1; index < numLayers; index++)
      {
         System.out.print("-" + layers[index]);
      }

      System.out.println();

      System.out.println("Saving state: " + saveWeights);

      if (isTraining)
      {
         System.out.println("Weight range: " + weightRangeStart + " - " + weightRangeEnd);
         System.out.println("Max iterations: " + maxIterations);
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Lambda value: " + lambda);
         System.out.println("Keep Alive Number: " + keepAlive);
      } // if (isTraining)

      System.out.println();

      System.out.println("Expected outputs path: " + expectedOutputsPath);
      System.out.println("Input file path: " + inputFilePath);
      System.out.println("Weights configuration: " + weightsConfig);

      if (saveWeights)
      {
         System.out.println("Saved weights path: " + savedWeightsPath);
      }
      else
      {
         System.out.println("Not saving weights.");
      } // if (saveWeights)
   } // public void printNetworkConfig(boolean isTraining)

   /**
    * Allocates memory for and initializes the instance variables, datasets, weight matrices, and
    * other arrays required for the network's operation.
    */
   public void memoryAllocation()
   {
      maxHiddenNodes = findMaxNodesInRange(INPUTLAYER+1, outputLayer-1);
      maxHiddenOutputNodes = Math.max(maxHiddenNodes, outputLayer);
      maxNodes = findMaxNodesInRange(INPUTLAYER, outputLayer);

      inputDataTable = new double[numTestCases][layers[INPUTLAYER]];
      outputDataTable = new double[numTestCases][layers[outputLayer]];
      outputs = new double[numTestCases][layers[outputLayer]];

      weights = new double[numLayers][maxNodes][maxNodes];
      activationLayers = new double[numLayers][maxNodes];

      if (isTraining)
      {
         contTraining = true;
         thetas = new double[numLayers][maxHiddenOutputNodes];
         psis = new double[numLayers][maxNodes];
      } // if (isTraining)
   } // public void memoryAllocation()

   /**
    * Finds and returns the max number of nodes within a given range.
    *
    * @param startLayer The starting layer of the range
    * @param endLayer   The last layer of the range
    * @return The max number of nodes within a range.
    */
   public int findMaxNodesInRange(int startLayer, int endLayer)
   {
      int maxNodes = 0;

      for (int layerIndex = startLayer; layerIndex <= endLayer; layerIndex++)
      {
         assert layers != null;

         if (layers[layerIndex] > maxNodes)
         {
            maxNodes = layers[layerIndex];
         }
      } // for (int layerIndex = startLayer; layerIndex <= endLayer; layerIndex++)

      return maxNodes;
   } // public int findMaxNodesInRange(int startLayer, int endLayer)

   /**
    * Populates all arrays of the network (input dataset, input-to-hidden and hidden-to-output weights)
    * with random values within the specified range or load them from a weights file.
    */
   public void populateArrays() throws FileNotFoundException
   {
      Scanner inputSc = new Scanner(new File(inputFilePath));
      Scanner testCaseSc = new Scanner(new File(expectedOutputsPath));

      switch (weightsConfig)
      {
         case "rand" ->
         {
            for (int n = HIDDENLAYER1; n < numLayers; n++)
            {
               for (int j = 0; j < layers[n-1]; j++)
               {
                  for (int k = 0; k < layers[n]; k++)
                  {
                     weights[n][j][k] = getRandomWeightInRange();
                  }
               }
            } // for (int n = HIDDENLAYER1; n < numLayers; n++)
         } // case "rand" ->

         case "load" -> loadWeights();

         case "zero" ->
         {
            for (int n = HIDDENLAYER1; n < numLayers; n++)
            {
               for (int j = 0; j < layers[n-1]; j++)
               {
                  for (int k = 0; k < layers[n]; k++)
                  {
                     weights[n][j][k] = 0;
                  }
               }
            } // for (int n = HIDDENLAYER1; n < numLayers; n++)
         } // case "zero" ->

         default -> throw new IllegalArgumentException("Weight configuration does not meet standards");
      } // switch (weightsConfig)

      scanFile(inputSc, layers[INPUTLAYER], inputDataTable);
      inputSc.close();

      scanFile(testCaseSc, layers[outputLayer], outputDataTable);
      testCaseSc.close();
   } // public void populateArrays() throws FileNotFoundException

   /**
    * Scans all inputs from the specified scanner and populate the given dataset.
    *
    * @param sc File's scanner
    */
   private void scanFile(Scanner sc, int nodes, double[][] data)
   {
      for (int index = 0; index < numTestCases; index++)
      {
         for (int k = 0; k < nodes; k++)
         {
            sc.nextLine();
            sc.nextLine();
            data[index][k] = sc.nextDouble();
         }
      } // for (int index = 0; index < numTestCases; index++)

      sc.close();
   } // private void scanFile(Scanner sc, int nodes, double[][] data)

   /**
    * Generates a random weight value within the specified range set during network configuration.
    *
    * @return A randomly generated weight value.
    */
   private double getRandomWeightInRange()
   {
      return Math.random() * (weightRangeEnd - weightRangeStart) + weightRangeStart;
   } // private double getRandomWeightInRange()

   /**
    * Saves the weights in a file named savedWeights.txt
    */
   public void saveWeights() throws FileNotFoundException
   {
      try (PrintWriter writer = new PrintWriter((savedWeightsPath)))
      {
         for (int n = HIDDENLAYER1; n < numLayers; n++)
         {
            for (int j = 0; j < layers[n-1]; j++)
            {
               for (int k = 0; k < layers[n]; k++)
               {
                  writer.println(weights[n][j][k]);
               }
            }
         } // for (int n = HIDDENLAYER1; n < numLayers; n++)
      } // try (PrintWriter writer = new PrintWriter((savedWeightsPath)))
   } // public void saveWeights() throws FileNotFoundException

   /**
    * Loads the weights from savedWeights.txt
    */
   public void loadWeights() throws FileNotFoundException
   {
      Scanner sc = new Scanner(new File(savedWeightsPath));

      for (int n = HIDDENLAYER1; n < numLayers; n++)
      {
         for (int j = 0; j < layers[n-1]; j++)
         {
            for (int k = 0; k < layers[n]; k++)
            {
               weights[n][j][k] = sc.nextDouble();
            }
         }
      } // for (int n = HIDDENLAYER1; n < numLayers; n++)

      sc.close();
   } // public void loadWeights() throws FileNotFoundException

   /**
    * Loads the input activation layer by setting its nodes to values of the input data table.
    *
    * @param index The current test case being run
    */
   public void loadInputActivations(int index)
   {
      for (int m = 0; m < layers[INPUTLAYER]; m++)
      {
         activationLayers[INPUTLAYER][m] = inputDataTable[index][m];
      } // for (int m = 0; m < layers[INPUTLAYER]; m++)
   } // public void loadInputActivations(int trainIndex)

   /**
    * Forward pass the training hidden and output activations.
    * Updates the activations based on dot product of weights and inputs.
    * Apply the activation function to the thetas to get output values.
    *
    * @param trainIndex The current test case that the training algorithm is on
    */
   public void forwardPass(int trainIndex)
   {
      double smallOmega;

      for (int n = HIDDENLAYER1; n < outputLayer; n++)
      {
         for (int k = 0; k < layers[n]; k++)
         {
            thetas[n][k] = 0.0;

            for (int j = 0; j < layers[n-1]; j++)
            {
               thetas[n][k] += activationLayers[n-1][j] * weights[n][j][k];
            }

            activationLayers[n][k] = activationFunction(thetas[n][k]);
         } // for (int k = 0; k < layers[n]; k++)
      } // for (int n = HIDDENLAYER1; n < outputLayer; n++)

      int n = outputLayer;
      double theta;

      for (int i = 0; i < layers[n]; i++)
      {
         theta = 0.0;  // zeroing the accumulator

         for (int j = 0; j < layers[n-1]; j++)
         {
            theta += activationLayers[n-1][j] * weights[n][j][i];
         } // for (int j = 0; j < layers[n-1]; j++)

         activationLayers[n][i] = activationFunction(theta);
         smallOmega = (outputDataTable[trainIndex][i] - activationLayers[n][i]);
         psis[n][i] = smallOmega * activationFunctionDerivative(theta);
         averageError += errorFunction(smallOmega);
      } // for (int i = 0; i < layers[n]; i++)
   } // public void forwardPass(int trainIndex)

   /**
    * Starts the training process using the dataset, target outputs, and training parameters previously set
    * It performs forward passes, computes errors, saves the errors, and
    * updates weights using gradient descent and back propagation until the specified conditions are met.
    */
   public void trainNetwork()
   {
      System.out.println();
      System.out.println("Training the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");
      System.out.println();

      long startTime = System.currentTimeMillis();
      double omegas;

      while (contTraining)
      {
         averageError = 0;

         for (int trainIndex = 0; trainIndex < numTestCases; trainIndex++)
         {
            loadInputActivations(trainIndex);

            forwardPass(trainIndex);

            for (int n = outputLayer - 1; n > HIDDENLAYER1; n--)
            {
               for (int k = 0; k < layers[n]; k++)
               {
                  omegas = 0.0; // zeroing the accumulator

                  for (int j = 0; j < layers[n+1]; j++)
                  {
                     omegas += psis[n+1][j] * weights[n+1][k][j];
                     weights[n+1][k][j] += lambda * activationLayers[n][k] * psis[n+1][j];
                  } // for (int j = 0; j < layers[n+1]; j++)

                  psis[n][k] = omegas * activationFunctionDerivative(thetas[n][k]);
               } // for (int k = 0; k < layers[n]; k++)
            } // for (int n = outputLayer - 1; n > HIDDENLAYER1; n--)

            int n = HIDDENLAYER1;
            for (int m = 0; m < layers[n]; m++)
            {
               omegas = 0.0;

               for (int k = 0; k < layers[n+1]; k++)
               {
                  omegas += psis[n+1][k] * weights[n+1][m][k];
                  weights[n+1][m][k] += lambda * activationLayers[n][m] * psis[n+1][k];
               }

               psis[n][m] = omegas * activationFunctionDerivative(thetas[n][m]);

               for (int x = 0; x < layers[INPUTLAYER]; x++)
               {
                  weights[n][x][m] += lambda * activationLayers[n-1][x] * psis[n][m];
               }
            } // for (int m = 0; m < layers[n]; m++)

            runningOneTestCase(trainIndex);

            n = outputLayer;
            for (int j = 0; j < layers[n]; j++)
            {
               outputs[trainIndex][j] = activationLayers[n][j];
            }
         } // for (int trainIndex = 0; trainIndex < numTestCases; trainIndex++)

         averageError /= (double) numTestCases;
         numIterations++;

         if (averageError <= errorThreshold)
         {
            errorThresholdReached = true;
            contTraining = false;
         } //if (averageError <= errorThreshold)

         if (numIterations >= maxIterations)
         {
            maxIterationsReached = true;
            contTraining = false;
         } //if (numIterations >= maxIterations)

         if (keepAlive != 0 && numIterations % keepAlive == 0)
         {
            System.out.printf("Iteration %d, Error = %f\n", numIterations, averageError);
         }
      } // while (contTraining)

      elapsed = System.currentTimeMillis() - startTime;

      System.out.println("Training completed.");
      System.out.println();
   } // public void trainNetwork()

   /**
    * Runs one test case through the network by forward passing it and applying already trained weights.
    *
    * @param runIndex The test case currently being run
    */
   public void runningOneTestCase(int runIndex)
   {
      double theta;

      for (int n = HIDDENLAYER1; n < numLayers; n++)
      {
         for (int k = 0; k < layers[n]; k++)
         {
            theta = 0.0;

            for (int m = 0; m < layers[n-1]; m++)
            {
               theta += weights[n][m][k] * activationLayers[n-1][m];                 //applying dot product
            }

            activationLayers[n][k] = activationFunction(theta);                      //applying activation function
         } // for (int k = 0; k < layers[n]; k++)
      } // for (int n = HIDDENLAYER1; n < numLayers; n++)
   } // public void runningOneTestCase(int runIndex)

   /**
    * Runs the network operation by forward passing all inputs and applying already trained weights
    */
   public void runNetwork()
   {
      System.out.println();
      System.out.println("Running the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");
      double theta;

      for (int runIndex = 0; runIndex < numTestCases; runIndex++)
      {
         loadInputActivations(runIndex);

         for (int n = HIDDENLAYER1; n < outputLayer; n++)
         {
            for (int j = 0; j < layers[n]; j++)
            {
               theta = 0.0;

               for (int k = 0; k < layers[n-1]; k++)
               {
                  theta += weights[n][k][j] * activationLayers[n-1][k];
               }

               activationLayers[n][j] = activationFunction(theta);
            } // for (int j = 0; j < layers[n]; j++)
         } // for (int n = HIDDENLAYER1; n < outputLayer; n++)

         int n = outputLayer;
         for (int i = 0; i < layers[n]; i++)
         {
            theta = 0.0;
            for (int j = 0; j < layers[n-1]; j++)
            {
               theta += weights[n][j][i] * activationLayers[n-1][j];
            }
            activationLayers[n][i] = activationFunction(theta);
            outputs[runIndex][i] = activationLayers[n][i];
         } // for (int i = 0; i < layers[n]; i++)
      } // for (int runIndex = 0; runIndex < numTestCases; runIndex++)

      System.out.println();
      System.out.println("Running Completed.");
   } // public static void runNetwork()

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
    * The activation function, used to calculate the output of neurons.
    *
    * @param input The input value to the neuron.
    * @return The output of the neuron after applying the activation function.
    */
   public double activationFunction(double input)
   {
      return sigmoid(input);
   }

   /**
    * Computes the derivative of the activation function, necessary for backpropagation.
    *
    * @param output The output of the neuron.
    * @return The derivative of the activation function at the given output value.
    */
   public double activationFunctionDerivative(double output)
   {
      return sigmoidDerivative(output);
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
      double sig = activationFunction(output);
      return sig * (1.0 - sig);
   }

   /**
    * Prints the results of the training process or running process.
    * If training, results include the error reached, iterations reached, and reason for an end of run.
    * for (int j = 0; j < numActs[outputLocation]; j++)
    *             {
    *                System.out.print(" " + String.format("%.0f", outputDataset[k][j]) + " ");
    *             }
    *             System.out.print("| - |");A truth table is outputted for both training and running.
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
         System.out.println("Average error: " + averageError);
      } // if (isTraining)
      System.out.println("Time Elapsed: " + elapsed + " ms");

      System.out.println();

      if (isTraining)
      {
         System.out.println("\t\tTraining Results");
      }
      else
      {
         System.out.println("\t\tRunning Results");
      } // if (isTraining)

      System.out.println("------------------------------------------------------");

      for (int colNum = 0; colNum < layers[INPUTLAYER]; colNum++)
      {
         System.out.print("I" + (colNum + 1) + " |");
      }

      System.out.println("       Expected Outputs            |          Actual Outputs           |");

      for (int index = 0; index < numTestCases; index++)
      {
         for (int colNum = 0; colNum < layers[INPUTLAYER]; colNum++)
         {
            System.out.print(inputDataTable[index][colNum] + "|");
         } // for (int colNum = 0; colNum < inputNodes; colNum++)

         for (int i = 0; i < layers[outputLayer]; i++)
         {
            DecimalFormat formatter = new DecimalFormat("0.000000000");
            System.out.print(formatter.format(outputDataTable[index][i]) + "|");
         } // for (int i = 0; i < outputNodes; i++)

         for (int i = 0; i < layers[outputLayer]; i++)
         {
            DecimalFormat formatter = new DecimalFormat("0.000000000");
            System.out.print(formatter.format(outputs[index][i]) + "|");
         } // for (int i = 0; i < outputNodes; i++)

         System.out.println();
      } // for (int index = 0; index < numTestCases; index++)

      System.out.println();
   } // public void reportResults()

   /**
    * The main method for the NLayer class.
    * It creates an instance of the network, configures it,
    * trains or runs it, and prints the results.
    * @param args Command line arguments
    */
   public static void main(String[] args) throws IOException
   {
      String control;
      if (args.length == 0)
      {
         control = DEFAULT_CONTROL_FILE;
      }
      else
      {
         control = args[0];
      } // if (args.length == 0)

      NLayer network = new NLayer();

      network.setNetworkConfig(control);
      network.memoryAllocation();
      network.populateArrays();
      network.printNetworkConfig(network.isTraining);

      if (network.isTraining)
      {
         network.trainNetwork();
         network.runNetwork();
      }
      else
      {
         network.runNetwork();
      } // if (network.isTraining)

      if (network.saveWeights)
      {
         network.saveWeights();
      } // if (network.saveWeights)

      network.reportResults();
   } // public static void main(String[] args) throws IOException
} // public class NLayer