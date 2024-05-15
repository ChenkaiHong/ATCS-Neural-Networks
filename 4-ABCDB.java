import java.io.*;
import java.lang.Math;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * The ABCDB class is a three-layer artificial neural network with a single hidden layer.
 * It is capable of learning logical functions (such as AND, OR, XOR) through training on a provided dataset
 * and uses a sigmoid activation function, gradient descent, and back propagation for weight optimization.
 *
 * @author Kai Hong
 *
 * Date of Creation: 04/11/2024
 */
public class ABCDB
{
   private static final String DEFAULT_CONTROL_FILE = "ABCDBControl.txt"; // Default control file if no argument is passed

   private int inputNodes;                                                // Number of nodes in the input layer
   private int hiddenKNodes;                                              // Num of nodes in the hidden k layer
   private int hiddenJNodes;                                              // Number of nodes in the hidden j layer
   private int outputNodes;                                               // Number of nodes in the output layer
   private final int NUMLAYERS = 4;                                       // Number of activation layers
   private int maxHiddenNodes;                                            // Max number of hidden nodes
   private int maxHiddenOutputNodes;                                      // Max number of hidden and output nodes
   private int maxNodes;                                                  // Max number of input, hidden, and output nodes

   private int numTestCases;                                              // Number of test cases

   private final int INPUTLAYER = 0;                                      // Index for the input layer
   private final int HIDDENLAYER1 = 1;                                    // Index for the hidden k layer
   private final int HIDDENLAYER2 = 2;                                    // Index for the hidden j layer
   private final int OUTPUTLAYER = 3;                                     // Index for the output layer
   private double weightRangeStart;                                       // Lower bound of the random weight range
   private double weightRangeEnd;                                         // Upper bound of the random weight range

   private int maxIterations;                                             // Maximum number of training iterations
   private double errorThreshold;                                         // Threshold for error termination
   private double lambda;                                                 // Learning rate for weight updates

   private double[][] thetas;                                             // 2-D array for all training theta values
   private double[][] activationLayers;                                   // 2-D array for all values in the activation layers
   private double[] smallPsis;                                            // Array for all small psi values
   private double[][] psis;                                               // 2-D array for all big psi values

   private double[][][] weights;                                          // Weights between input and hidden layer
   private boolean isTraining;                                            // Indicates if the network is in training mode
   private boolean contTraining;                                          // Indicates if the network should keep training

   private double[][] inputDataTable;                                     // Input dataset
   private double[][] outputDataTable;                                    // Output dataset
   private double[][] outputs;                                            // Final outputs of the network
   private double averageError;                                           // Average error for the network
   private int numIterations;                                             // Tracks the number of iterations

   private boolean errorThresholdReached;                                 // Indicates if the error threshold was reached
   private boolean maxIterationsReached;                                  // Indicates if max iterations limit was reached

   private long elapsed;                                                  // Time it took for network to run or train
   private String expectedOutputsPath;                                    // Absolute file path of the expected outputs file
   private String inputFilePath;                                          // Absolute file path of the inputs file

   private String weightsConfig;                                          // Configuration of weights
   private boolean saveWeights;                                           // Indicates if weights are saved
   private String savedWeightsPath;                                       // Absolute path of the weights file

   /**
    * Configures the neural network with specified parameters.
    */
   public void setNetworkConfig(String controlFilePath) throws IOException
   {
      Scanner sc = new Scanner(new File("/Users/kh/Desktop/Shared/ABCDB/src/" + controlFilePath));

      sc.nextLine();
      isTraining = sc.nextBoolean();
      sc.nextLine();
      sc.nextLine();

      inputNodes = sc.nextInt();
      sc.nextLine();
      sc.nextLine();
      hiddenKNodes = sc.nextInt();
      sc.nextLine();
      sc.nextLine();
      hiddenJNodes = sc.nextInt();
      sc.nextLine();
      sc.nextLine();
      outputNodes = sc.nextInt();
      sc.nextLine();
      sc.nextLine();

      maxHiddenNodes = Math.max(hiddenKNodes, hiddenJNodes);
      maxHiddenOutputNodes = Math.max(maxHiddenNodes, outputNodes);
      maxNodes = Math.max(inputNodes, maxHiddenOutputNodes);

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
   } // public void setNetworkConfig()


   /**
    * Prints the current configuration of the neural network to the console.
    *
    * @param isTraining Indicates whether the network is currently in training mode.
    */
   public void printNetworkConfig(boolean isTraining)
   {
      System.out.println("Network Configuration: " + inputNodes + "-" + hiddenKNodes + "-" + hiddenJNodes + "-" + outputNodes);
      System.out.println("Saving state: " + saveWeights);

      if (isTraining)
      {
         System.out.println("Weight range: " + weightRangeStart + " - " + weightRangeEnd);
         System.out.println("Max iterations: " + maxIterations);
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Lambda value: " + lambda);
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
      inputDataTable = new double[numTestCases][inputNodes];
      outputDataTable = new double[numTestCases][outputNodes];
      outputs = new double[numTestCases][outputNodes];
      weights = new double[numTestCases][maxNodes][maxNodes];
      activationLayers = new double[NUMLAYERS][maxNodes];

      if (isTraining)
      {
         contTraining = true;
         thetas = new double[NUMLAYERS][maxHiddenOutputNodes];
         smallPsis = new double[outputNodes];
         psis = new double[NUMLAYERS][maxHiddenNodes];
      } //if (isTraining)
   } // public void memoryAllocation()


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
             for (int m = 0; m < inputNodes; m++)
             {
                for (int k = 0; k < hiddenKNodes; k++)
                {
                   weights[HIDDENLAYER1][m][k] = getRandomWeightInRange();
                }
             } //  for (int m = 0; m < inputNodes; m++)
             for (int k = 0; k < hiddenKNodes; k++)
             {
                for (int j = 0; j < hiddenJNodes; j++)
                {
                   weights[HIDDENLAYER2][k][j] = getRandomWeightInRange();
                }
             } // for (int k = 0; k < hiddenKNodes; k++)
             for (int j = 0; j < hiddenJNodes; j++)
             {
                for (int i = 0; i < outputNodes; i++)
                {
                   weights[OUTPUTLAYER][j][i] = getRandomWeightInRange();
                }
             } // for (int j = 0; j < hiddenJNodes; j++)
          } // case "rand" ->
          case "load" -> loadWeights();
          case "zero" ->
          {
             for (int m = 0; m < inputNodes; m++)
             {
                for (int k = 0; k < hiddenKNodes; k++)
                {
                   weights[HIDDENLAYER1][m][k] = 0;
                }
             } // for (int m = 0; m < inputNodes; m++)
             for (int k = 0; k < hiddenKNodes; k++)
             {
                for (int j = 0; j < hiddenJNodes; j++)
                {
                   weights[HIDDENLAYER2][k][j] = 0;
                }
             } // for (int k = 0; k < hiddenKNodes; k++)

             for (int j = 0; j < hiddenJNodes; j++)
             {
                for (int i = 0; i < outputNodes; i++)
                {
                   weights[OUTPUTLAYER][j][i] = 0;
                }
             } // for (int j = 0; j < hiddenJNodes; j++)
          } // case "zero" ->
          default -> throw new IllegalArgumentException("Weight configuration does not meet standards");
       } // switch (weightsConfig)

      scanFile(inputSc, inputNodes, inputDataTable);
      inputSc.close();

      scanFile(testCaseSc, outputNodes, outputDataTable);
      testCaseSc.close();
   } // public void populateArrays()

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
   }

   /*
    * Saves the weights in a file named savedWeights.txt
    */
   public void saveWeights() throws FileNotFoundException
   {
      try (PrintWriter writer = new PrintWriter((savedWeightsPath)))
      {
         for (int m = 0; m < inputNodes; m++)
         {
            for (int k = 0; k < hiddenKNodes; k++)
            {
               writer.println(weights[HIDDENLAYER1][m][k]);
            }
         } // for (int m = 0; m < inputNodes; m++)

         for (int k = 0; k < hiddenKNodes; k++)
         {
            for (int j = 0; j < hiddenJNodes; j++)
            {
               writer.println(weights[HIDDENLAYER2][k][j]);
            }
         } // for (int k = 0; k < hiddenKNodes; k++)

         for (int j = 0; j < hiddenJNodes; j++)
         {
            for (int i = 0; i < outputNodes; i++)
            {
               writer.println(weights[OUTPUTLAYER][j][i]);
            }
         } // for (int j = 0; j < hiddenJNodes; j++)
      } // PrintWriter writer = new PrintWriter((savedWeightsPath))
   } //public void saveWeights() throws FileNotFoundException

   /*
    * Loads the weights from savedWeights.txt
    */
   public void loadWeights() throws FileNotFoundException
   {
      Scanner sc = new Scanner(new File(savedWeightsPath));

      for (int m = 0; m < inputNodes; m++)
      {
         for (int k = 0; k < hiddenKNodes; k++)
         {
            weights[HIDDENLAYER1][m][k] = sc.nextDouble();
         }
      } // for (int m = 0; m < inputNodes; m++)

      for (int k = 0; k < hiddenKNodes; k++)
      {
         for (int j = 0; j < hiddenJNodes; j++)
         {
            weights[HIDDENLAYER2][k][j] = sc.nextDouble();
         }
      } // for (int k = 0; k < hiddenKNodes; k++)

      for (int j = 0; j < hiddenJNodes; j++)
      {
         for (int i = 0; i < outputNodes; i++)
         {
            weights[OUTPUTLAYER][j][i] = sc.nextDouble();
         }
      } // for (int j = 0; j < hiddenJNodes; j++)

      sc.close();
   } //public void loadWeights() throws FileNotFoundException

   /**
    * Forward pass the training hidden and output activations.
    * Updates the activations based on dot product of weights and inputs.
    * Apply the activation function to the thetas to get output values.
    */
   public void forwardPass(int trainIndex)
   {
      double smallOmega;
      int n; // index for each activation layer

      n = HIDDENLAYER1;
      for (int k = 0; k < hiddenKNodes; k++)
      {
         thetas[n][k] = 0.0; // zeroing the accumulator

         for (int m = 0; m < inputNodes; m++)
         {
            thetas[n][k] += activationLayers[INPUTLAYER][m] * weights[n][m][k];
         }

         activationLayers[n][k] = sigmoid(thetas[n][k]);
      } // for (int k = 0; k < hiddenKNodes; k++)

      n = HIDDENLAYER2;
      for (int j = 0; j < hiddenJNodes; j++)
      {
         thetas[n][j] = 0.0;

         for (int k = 0; k < hiddenKNodes; k++)
         {
            thetas[n][j] += activationLayers[HIDDENLAYER1][k] * weights[n][k][j];
         }

         activationLayers[n][j] = sigmoid(thetas[n][j]);
      } // for (int j = 0; j < hiddenJNodes; j++)

      n = OUTPUTLAYER;
      for (int i = 0; i < outputNodes; i++)
      {
         thetas[n][i] = 0.0;  // zeroing the accumulator

         for (int j = 0; j < hiddenJNodes; j++)
         {
            thetas[n][i] += activationLayers[HIDDENLAYER2][j] * weights[n][j][i];
         } // for (int j = 0; j < hiddenJNodes; j++)

         activationLayers[n][i] = sigmoid(thetas[n][i]);
         smallOmega = (outputDataTable[trainIndex][i] - activationLayers[n][i]);
         smallPsis[i] = smallOmega * sigmoidDerivative(thetas[n][i]);
         averageError += errorFunction(outputDataTable[trainIndex][i] - activationLayers[n][i]);
      } // for (int i = 0; i < outputNodes; i++)
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
            int n; // index for each activation layer

            n = INPUTLAYER;
            for (int m = 0; m < inputNodes; m++)
            {
               activationLayers[n][m] = inputDataTable[trainIndex][m];
            } // for (int m = 0; m < inputNodes; m++)

            forwardPass(trainIndex);

            n = HIDDENLAYER2;
            for (int j = 0; j < hiddenJNodes; j++)
            {
               omegas = 0.0; // zeroing the accumulator

               for (int i = 0; i < outputNodes; i++)
               {
                  omegas += smallPsis[i] * weights[OUTPUTLAYER][j][i];
                  weights[OUTPUTLAYER][j][i] += lambda * activationLayers[n][j] * smallPsis[i];
               } // for (int i = 0; i < outputNodes; i++)

               psis[n][j] = omegas * sigmoidDerivative(thetas[n][j]);
            } // for (int j = 0; j < hiddenJNodes; j++)

            n = HIDDENLAYER1;
            for (int k = 0; k < hiddenKNodes; k++)
            {
               omegas = 0.0;

               for (int j = 0; j < hiddenJNodes; j++)
               {
                  omegas += psis[HIDDENLAYER2][j] * weights[HIDDENLAYER2][k][j];
                  weights[HIDDENLAYER2][k][j] += lambda * activationLayers[n][k] * psis[HIDDENLAYER2][j];
               }

               psis[n][k] = omegas * sigmoidDerivative(thetas[n][k]);

               for (int m = 0; m < inputNodes; m++)
               {
                  weights[n][m][k] += lambda * activationLayers[INPUTLAYER][m] * psis[n][k];
               }
            } // for (int k = 0; k < hiddenKNodes; k++)

            runningOneTestCase(trainIndex);
         } // for (int trainIndex = 0; trainIndex < numTestCases; trainIndex++)

         averageError /= numTestCases;
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
      } // while (contTraining)

      elapsed = System.currentTimeMillis() - startTime;

      System.out.println("Training completed.");
      System.out.println();
   } // public void trainNetwork()

   /**
    * Runs one test case through the network by forward passing it and applying already trained weights
    *
    * @param runIndex The test case being run
    */
   public void runningOneTestCase(int runIndex)
   {
      double theta;
      int n;           // index for each activation layer

      n = HIDDENLAYER1;
      for (int k = 0; k < hiddenKNodes; k++)
      {
         theta = 0.0;  // zeroing the accumulator

         for (int m = 0; m < inputNodes; m++)
         {
            theta += activationLayers[INPUTLAYER][m] * weights[n][m][k];
         }

         activationLayers[n][k] = sigmoid(theta);
      } // for (int k = 0; k < hiddenKNodes; k++)

      n = HIDDENLAYER2;
      for (int j = 0; j < hiddenJNodes; j++)
      {
         theta = 0.0;  // zeroing the accumulator

         for (int k = 0; k < hiddenKNodes; k++)
         {
            theta += activationLayers[HIDDENLAYER1][k] * weights[n][k][j];
         } // for (int j = 0; j < hiddenNodes; j++)

         activationLayers[n][j] = sigmoid(theta);
      } // for (int j = 0; j < hiddenJNodes; j++)

      n = OUTPUTLAYER;
      for (int i = 0; i < outputNodes; i++)
      {
         theta = 0.0;  // zeroing the accumulator

         for (int j = 0; j < hiddenJNodes; j++)
         {
            theta += activationLayers[HIDDENLAYER2][j] * weights[n][j][i];
         } // for (int j = 0; j < hiddenJNodes; j++)

         activationLayers[n][i] = sigmoid(theta);
         outputs[runIndex][i] = activationLayers[n][i];
      } // for (int i = 0; i < outputNodes; i++)
   } // public void runningOneTestCase(int testCase)

   /**
    * Runs the network operation by forward passing all inputs and applying already trained weights
    */
   public void runNetwork()
   {
      System.out.println();
      System.out.println("Running the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");

      for (int runIndex = 0; runIndex < numTestCases; runIndex++)
      {
         int n; // index for input activation layer
         for (int m = 0; m < inputNodes; m++)
         {
            n = INPUTLAYER;
            activationLayers[n][m] = inputDataTable[runIndex][m];
         } // for (int m = 0; m < inputNodes; m++)

         runningOneTestCase(runIndex);
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

      System.out.println("--------------------------------------------");

      for (int colNum = 0; colNum < inputNodes; colNum++)
      {
         System.out.print("I" + (colNum + 1) + " |");
      }

      System.out.println("    AND    |     OR    |    XOR    |");

      for (int index = 0; index < numTestCases; index++)
      {
         for (int colNum = 0; colNum < inputNodes; colNum++)
         {
            System.out.print(inputDataTable[index][colNum] + "|");
         } // for (int colNum = 0; colNum < inputNodes; colNum++)

         for (int i = 0; i < outputNodes; i++)
         {
            DecimalFormat formatter = new DecimalFormat("0.000000000");
            System.out.print(formatter.format(outputs[index][i]) + "|");
         } // for (int i = 0; i < outputNodes; i++)

         System.out.println();
      } // for (int index = 0; index < numTestCases; index++)

      System.out.println();
   } // public void reportResults()

   /**
    * The main method for the ABCDB class.
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

      ABCDB network = new ABCDB();

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
} // public class ABCDB
