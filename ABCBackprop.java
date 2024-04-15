import java.io.*;
import java.lang.Math;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * The ABCBackprop class is a three-layer artificial neural network with a single hidden layer.
 * It is capable of learning logical functions (such as AND, OR, XOR) through training on a provided dataset
 * and uses a sigmoid activation function, gradient descent, and back propagation for weight optimization.
 *
 * @author Kai Hong
 *
 * Date of Creation: 03/11/2024
 */
public class ABCBackprop
{
   private static final String DEFAULT_CONTROL_FILE = "ABCControl.txt"; // Default control file if no argument is passed

   private int inputNodes;                                              // Number of nodes in the input layer
   private int hiddenNodes;                                             // Number of nodes in the hidden layer
   private int outputNodes;                                             // Number of nodes in the output layer

   private int numTestCases;                                            // Number of test cases

   private double[] hiddenLayer;                                        // Values stored in the hidden layer

   private double weightRangeStart;                                     // Lower bound of the random weight range
   private double weightRangeEnd;                                       // Upper bound of the random weight range

   private int maxIterations;                                           // Maximum number of training iterations
   private double errorThreshold;                                       // Threshold for error termination
   private double lambda;                                               // Learning rate for weight updates

   private double[] thetas;                                             // 2-D array for all training theta values
   private double[] smallPsis;                                          // Array for all small psi values

   private double[][] inputToHiddenWeights;                             // Weights between input and hidden layer
   private double[][] hiddenToOutputWeights;                            // Weights between hidden and output layer
   private boolean isTraining;                                          // Indicates if the network is in training mode
   private boolean contTraining;                                        // Indicates if the network should keep training

   private double[][] inputDataTable;                                   // Input dataset
   public double[][] outputDataTable;                                   // output dataset

   private double[] outputValues;                                       // Outputs of each run

   private double[][] outputs;                                          // Final outputs of the network
   private double averageError;                                         // Average error for the network
   private int numIterations;                                           // Tracks the number of iterations

   private boolean errorThresholdReached;                               // Indicates if the error threshold was reached
   private boolean maxIterationsReached;                                // Indicates if max iterations limit was reached

   private long elapsed;                                                // Time it took for network to run or train
   private String networkConfig;                                        // Configuration of the network
   private String expectedOutputsPath;                                  // Absolute file path of the expected outputs file
   private String inputFilePath;                                        // Absolute file path of the inputs file

   private String weightsConfig;                                        // Configuration of weights
   private boolean saveWeights;                                         // Indicates if weights are saved
   private String savedWeightsPath;                                     // Absolute path of the weights file

   /**
    * Configures the neural network with specified parameters.
    */
   public void setNetworkConfig(String controlFilePath) throws IOException
   {
      Scanner sc = new Scanner(new File("/Users/kh/Desktop/Shared/ABCBackprop/src/" + controlFilePath));

      sc.nextLine();
      isTraining = sc.nextBoolean();
      sc.nextLine();
      sc.nextLine();

      inputNodes = sc.nextInt();
      sc.nextLine();
      sc.nextLine();
      hiddenNodes = sc.nextInt();
      sc.nextLine();
      sc.nextLine();
      outputNodes = sc.nextInt();
      sc.nextLine();
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

      networkConfig = sc.next();
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

      return;
   } // public void setNetworkConfig()


   /**
    * Prints the current configuration of the neural network to the console.
    *
    * @param isTraining Indicates whether the network is currently in training mode.
    */
   public void printNetworkConfig(boolean isTraining)
   {
      System.out.println("Network Configuration: " + inputNodes + "-" + hiddenNodes + "-" + outputNodes);
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
      outputs = new double[numTestCases][hiddenNodes];

      inputToHiddenWeights = new double[inputNodes][hiddenNodes];
      hiddenToOutputWeights = new double[hiddenNodes][outputNodes];

      outputValues = new double[outputNodes];
      hiddenLayer = new double[hiddenNodes];

      if (isTraining)
      {
         contTraining = true;
         thetas = new double[hiddenNodes];
         smallPsis = new double[outputNodes];
      } //if (isTraining)
   } // public void memoryAllocation()


   /**
    * Populates all arrays of the network (input dataset, input-to-hidden and hidden-to-output weights)
    * with random values within the specified range.
    */
   public void populateArrays() throws FileNotFoundException
   {
      Scanner inputSc = new Scanner(new File(inputFilePath));
      Scanner testCaseSc = new Scanner(new File(expectedOutputsPath));

       switch (weightsConfig)
       {
           case "random" ->
           {
               for (int k = 0; k < inputNodes; k++)
               {
                   for (int j = 0; j < hiddenNodes; j++)
                   {
                       inputToHiddenWeights[k][j] = getRandomWeightInRange();
                   }
               } // for (int k = 0; k < inputNodes; k++)

               for (int j = 0; j < hiddenNodes; j++)
               {
                   for (int i = 0; i < outputNodes; i++)
                   {
                       hiddenToOutputWeights[j][i] = getRandomWeightInRange();
                   }
               } // for (int k = 0; k < hiddenNodes; k++)
           } // case "random" ->
           case "load" -> loadWeights();
           case "zero" ->
           {
               for (int k = 0; k < inputNodes; k++)
               {
                   for (int j = 0; j < hiddenNodes; j++)
                   {
                       inputToHiddenWeights[k][j] = 0;
                   }
               } // for (int k = 0; k < inputNodes; k++)

               for (int j = 0; j < hiddenNodes; j++)
               {
                   for (int i = 0; i < outputNodes; i++)
                   {
                       hiddenToOutputWeights[j][i] = 0;
                   }
               } // for (int j = 0; j < hiddenNodes; j++)
           } // case "zero" ->
           default -> throw new IllegalArgumentException("Weight configuration does not meet standards");
       } // switch (weightsConfig)

      scanInputFile(inputSc);
      inputSc.close();

      if (networkConfig.equals("ABC") || networkConfig.equals("3BC"))
      {
         for (int index = 0; index < numTestCases; index++)
         {
            for (int k = 0; k < outputNodes; k++)
            {
               testCaseSc.nextLine();
               testCaseSc.nextLine();
               outputDataTable[index][k] = testCaseSc.nextDouble();
            }
         } // for (int index = 0; index < numTestCases; index++)

         testCaseSc.close();
      } // if (networkConfig.equals("ABC") || networkConfig.equals("3BC"))
      else if (networkConfig.equals("AB1"))
      {
         testCaseSc.nextLine();
         testCaseSc.nextLine();

         outputDataTable[0][0] = testCaseSc.nextDouble();
         testCaseSc.nextLine();
         testCaseSc.nextLine();
         outputDataTable[1][0] = testCaseSc.nextDouble();
         testCaseSc.nextLine();
         testCaseSc.nextLine();
         outputDataTable[2][0] = testCaseSc.nextDouble();
         testCaseSc.nextLine();
         testCaseSc.nextLine();
         outputDataTable[3][0] = testCaseSc.nextDouble();

         testCaseSc.close();
      } // else if (networkConfig.equals("AB1"))

   } // public void populateArrays()

   /**
    * Scans all inputs from the inputSc scanner and populate the inputDataTable.
    *
    * @param inputSc Input file's scanner
    */
   private void scanInputFile(Scanner inputSc)
   {
      for (int index = 0; index < numTestCases; index++)
      {
         for (int k = 0; k < inputNodes; k++)
         {
            inputSc.nextLine();
            inputSc.nextLine();
            inputDataTable[index][k] = inputSc.nextDouble();
         }
      } // for (int index = 0; index < numTestCases; index++)
      inputSc.close();
   } // private void scanInputFile(Scanner inputSc)

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
      try (PrintWriter writer = new PrintWriter(new File(savedWeightsPath)))
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
   } //public void saveWeights() throws FileNotFoundException


   /*
    * Loads the weights from savedWeights.txt
    */
   public void loadWeights() throws FileNotFoundException
   {
      Scanner sc = new Scanner(new File(savedWeightsPath));

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
    * Runs the network operation by forward passing all inputs and applying already trained weights
    */
   public void runNetwork()
   {
      System.out.println();
      System.out.println("Running the network...");
      System.out.println();
      System.out.println("--------------------------------------------------------------------------------");

      double theta;

      long startTime = System.currentTimeMillis();

      for (int runIndex = 0; runIndex < numTestCases; runIndex++)
      {
         runningOneTestCase(runIndex);
      } // for (int runIndex = 0; runIndex < numTestCases; runIndex++)

      elapsed = System.currentTimeMillis() - startTime;

      System.out.println();
      System.out.println("Running Completed.");
   } // public static void runNetwork()

   /**
    * Runs one test case through the network by forward passing it and applying already trained weights
    *
    * @param testCase The test case being run
    */
   public void runningOneTestCase(int testCase)
   {
      double theta;

      for (int j = 0; j < hiddenNodes; j++)
      {
         theta = 0.0;  // zeroing the accumulator

         for (int k = 0; k < inputNodes; k++)
         {
            theta += inputDataTable[testCase][k] * inputToHiddenWeights[k][j];
         }

         hiddenLayer[j] = sigmoid(theta);
      } // for (int j = 0; j < hiddenNodes; j++)

      for (int i = 0; i < outputNodes; i++)
      {
         theta = 0.0;  // zeroing the accumulator

         for (int j = 0; j < hiddenNodes; j++)
         {
            theta += hiddenLayer[j] * hiddenToOutputWeights[j][i];
         } // for (int j = 0; j < hiddenNodes; j++)
         outputValues[i] = sigmoid(theta);
         outputs[testCase][i] = outputValues[i];
      } // for (int i = 0; i < outputNodes; i++)
   }

   /**
    * Forward pass the training hidden and output activations.
    * Updates the activations based on dot product of weights and inputs.
    * Applies the activation function to the thetas to get output values.
    */
   public void forwardPass(int trainIndex)
   {
      double theta;

      for (int j = 0; j < hiddenNodes; j++)
      {
         thetas[j] = 0.0; // zeroing the accumulator

         for (int k = 0; k < inputNodes; k++)
         {
            thetas[j] += inputDataTable[trainIndex][k] * inputToHiddenWeights[k][j];
         }

         hiddenLayer[j] = sigmoid(thetas[j]);
      } // for (int j = 0; j < hiddenNodes; j++)

      for (int i = 0; i < outputNodes; i++)
      {
         theta = 0.0;  // zeroing the accumulator

         for (int j = 0; j < hiddenNodes; j++)
         {
            theta += hiddenLayer[j] * hiddenToOutputWeights[j][i];
         } // for (int j = 0; j < hiddenNodes; j++)

         outputValues[i] = sigmoid(theta);
         averageError += errorFunction(outputDataTable[trainIndex][i] - outputValues[i]);
         smallPsis[i] = (outputDataTable[trainIndex][i] - outputValues[i]) * sigmoidDerivative(theta);
      } // for (int i = 0; i < outputNodes; i++)
   }

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

      double omegas;
      double psi;

      long startTime = System.currentTimeMillis();

      while (contTraining)
      {
         averageError = 0;

         for (int trainIndex = 0; trainIndex < numTestCases; trainIndex++)
         {
            forwardPass(trainIndex);

            for (int j = 0; j < hiddenNodes; j++)
            {
               omegas = 0.0; // zeroing the accumulator

               for (int i = 0; i < outputNodes; i++)
               {
                  omegas += smallPsis[i] * hiddenToOutputWeights[j][i];
                  hiddenToOutputWeights[j][i] += lambda * hiddenLayer[j] * smallPsis[i];;
               } // for (int i = 0; i < outputNodes; i++)

               psi = omegas * sigmoidDerivative(thetas[j]);

               for (int k = 0; k < inputNodes; k++)
               {
                  inputToHiddenWeights[k][j] += lambda * inputDataTable[trainIndex][k] * psi;
               } // for (int k = 0; k < inputNodes; k++)
            } // for (int j = 0; j < hiddenNodes; j++)

            runningOneTestCase(trainIndex);
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
      } // while (contTraining)

      elapsed = System.currentTimeMillis() - startTime;

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
    * The main method for the ABCBackprop class.
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

      ABCBackprop network = new ABCBackprop();
      network.setNetworkConfig(control);
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
      if (network.saveWeights)
      {
         network.saveWeights();
      }
       network.reportResults();
   } // public static void main(String args[])
} // public class ABCBackprop