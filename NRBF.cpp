#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <ctime>


////////////////////////////////////////////////////////////////////////
//																	  //
//			NORMALISED RADIAL BASIS FUNCTION NEURAL NETWORK			  //
//																	  //
//	    Joel Gooch, BSc Computer Science, University of Plymouth	  //
//																	  //
////////////////////////////////////////////////////////////////////////

//global variables
int maxEpochs = 10;
float rbfSigma = 0.6f;
float learningRate = 1.0f; 
float tolerance = 0.9f;

// class for individual data entries
class dataEntry {
public: 
	std::vector<float> features;
	float expectedClassification;
	dataEntry(std::vector<float> f, float c): features(f), expectedClassification(c) {}
};

// class to hold whole data set, providing embedded data splits
class dataSet {
public: 
	std::vector<dataEntry> trainingSet;
	std::vector<dataEntry> testingSet;
	std::vector<dataEntry> validationSet;
	dataSet() {}
};

// class used to read data from .txt file
class dataReader {
public: 
	dataSet dataSet;
	bool loadDataFile(const char* filename);
	void processLine(std::string line);
private:
	std::vector<dataEntry> data;
	int noOfFeatures;
	int noOfTargets;

};

// function that handles loading data from .txt file
bool dataReader::loadDataFile(const char* filename) {
	// initialise variable for tracking number of entries
	int noOfDataEntries = 0;

	// open file to read from
	std::fstream inputFile;
	inputFile.open(filename, std::ios::in);

	if(inputFile.is_open()) {
		std::string line = "";
		//read data from file
		while (!inputFile.eof()) {
			getline(inputFile, line);
			// check line is something other than just a blank new line
			if (line.length() > 2) {
				// check if the second from last character in line is a - sign
				// dynamically calculate number of features accordingly
				if (line.end()[-2] == '-') {
					noOfFeatures = line.size() / 2 - 1;
				} else {
					noOfFeatures = line.size() / 2;
				}
				//process individual line
				processLine(line);
				noOfDataEntries++;
			}
		}

		// randomize data order
		std::srand(std::time(0));
		random_shuffle(data.begin(), data.end());

		// calculate data index splits for given data
		//int trainingDataEndIndex = (int) (0.6 * noOfDataEntries);
		//int testingSetSize = (int)(ceil(0.2 * noOfDataEntries));
		//int validationSetSize = (int) (data.size() - trainingDataEndIndex - testingSetSize);

		// customise data index splits from whole data set
		int trainingDataEndIndex = (int)(0.6 * noOfDataEntries);
		int testingSetSize = noOfDataEntries - trainingDataEndIndex;
		int validationSetSize = 0;
		
		//fill training data set
		for (int i = 0; i < trainingDataEndIndex ; i++) {
			dataSet.trainingSet.push_back(data[i]);
		}

		// fill testing data set
		for (int i = trainingDataEndIndex; i < trainingDataEndIndex + testingSetSize; i++) {
			dataSet.testingSet.push_back(data[i]);
		}

		// fill validation data set
		for (int i = trainingDataEndIndex + testingSetSize; i < (int) data.size(); i++) {
			dataSet.validationSet.push_back(data[i]);
		}
		printf("success opening input file: %s, reads: %d \n", filename, data.size());

		// close file
		inputFile.close();
		return true;
	} else {
		printf("error opening input file: %s \n", filename);
		return false;
	}
}

// function to process individual line from .txt file
void dataReader::processLine(std::string line) {
	// initialise new data entry variables
	std::vector<float> features;
	float expectedClassification = 0;

	// store inputs
	char* cstr = new char[line.size()+1];
	char* t;
	strcpy_s(cstr, line.size() + 1, line.c_str());

	// tokenise 
	int i = 0;
	char* nextToken = NULL;
	t = strtok_s(cstr, ",", &nextToken);

	while (t != NULL && i < noOfFeatures + 1) {
		// allocate memory for new value
		float *value = (float*)malloc(sizeof(float));
		// convert string to float
		*value = std::stof(t);
		// add value to features or classification output
		if (i < noOfFeatures) {
			features.push_back(*value);
		} else { 
			expectedClassification = *value;
		}
		// move to next token
		t = strtok_s(NULL, ",", &nextToken);
		i++;
	}
	// add to data structure
	data.push_back(dataEntry(features, expectedClassification));
}

// basic definition for Connection struct
struct Connection {
	float weight;
};

// pre define Node class to allow Layer to be defined
class Node;

// typedef Layer class
typedef std::vector<Node> Layer;

// Node class definition
class Node {
public:
	Node(int numOutputs, int myIndex, std::vector<dataEntry> trainingSet, int layerNo);
	void setOutputVal(float val) {nodeOutputVal = val;}
	float getOutputVal(void) const {return nodeOutputVal;}
	void calculateWeightedSum(const Layer prevLayer);
	void gaussianRadialBasisFunction(const Layer prevLayer);
	std::vector<Connection> nodeOutputWeights;
private: 
	float nodeOutputVal;
	int nodeIndex;
};

// Node constructor
Node::Node(int numOutputs, int myIndex, std::vector<dataEntry> trainingData, int layerNum) {
	nodeIndex = myIndex;
	for (int connection = 0; connection < numOutputs; connection++) {
		// add new connection from node
		nodeOutputWeights.push_back(Connection());

		// pathway for constructing node in input layer
		if (layerNum == 0) {
			dataEntry entry = trainingData[connection];
			nodeOutputWeights.back().weight = entry.features[nodeIndex];
		// pathway for constructing node in hidden layer
		} else if (layerNum == 1) {
			dataEntry entry = trainingData[myIndex];
			nodeOutputWeights.back().weight = entry.expectedClassification;
		}
	}
}

// weighted sum calculation function for hidden-output layer calculation
void Node::calculateWeightedSum(const Layer prevLayer) {
	// initialise local variables for calculations
	float prevLayerSum = 0.0f;
	float totalSum = 0.0f;

	// for all nodes in layer
	for (unsigned node = 0; node < prevLayer.size(); node++) {
		// calculate weighted sums
		prevLayerSum += prevLayer[node].getOutputVal();
		totalSum += prevLayer[node].getOutputVal() * prevLayer[node].nodeOutputWeights[nodeIndex].weight;
	}
	totalSum /= prevLayerSum; // FOR NORMALISED RBF, COMMENT OUT FOR STANDARD RBF
	// set output value of node to summation value
	nodeOutputVal = totalSum;
}

// apply gaussian radial basis function for input-hidden layer calculation
void Node::gaussianRadialBasisFunction(const Layer prevLayer) {
	float totalSum = 0.0f;

	// cycle all nodes in layer
	for (unsigned node = 0; node < prevLayer.size(); node++) {
		// calculate sum of (input - weight)^2 for all nodes
		totalSum += pow(prevLayer[node].getOutputVal() - prevLayer[node].nodeOutputWeights[nodeIndex].weight, 2);
	}
	// divide total by 2sigma^2 and apply exponential function
	nodeOutputVal = exp(-(totalSum / (2 * pow(rbfSigma, 2))));
} 

// function used to print results of feed forward
void printResults(int numMisclassified, int numPoisonousMisclassified, int numEdibleMisclassified) {
	printf("%d mushrooms misclassified \n", numMisclassified);
	printf("%d should have been classified as poisonous but were classified as edible \n", numPoisonousMisclassified);
	printf("%d should have been classified as edible but were classified as poisonous \n \n", numEdibleMisclassified);
}

// Network class definition 
class Network {
public:
	Network(const std::vector <int> topology, std::vector<dataEntry> trainingData);
	void feedForward(std::vector<float> inputVals);
	void trainNetwork(std::vector<dataEntry> trainingData, std::vector<dataEntry> testingData);
	std::vector<Layer> networkLayers;
};

// Network constructor
Network::Network(const std::vector <int> topology, std::vector<dataEntry> trainingData) {
	int numLayers = topology.size();
	for (int layerNum = 0; layerNum < numLayers; layerNum++) {
		// create new layer
		networkLayers.push_back(Layer());
		// if layernum is output layer then number of outputs is 0, 
		// otherwise number of outputs is number of nodes in next layer
		int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		// add all required nodes for specified layer
		for (int node = 0; node < topology[layerNum]; node++) {
			networkLayers.back().push_back(Node(numOutputs, node, trainingData, layerNum));
		}
	}
}

// function that feeds input values through network and calculates output
void Network::feedForward(std::vector<float> inputVals) {

	// set output values of input nodes
	for (unsigned i = 0; i < inputVals.size(); i++) {
		networkLayers[0][i].setOutputVal(inputVals[i]);
	}

	Layer inputLayer = networkLayers[0];
	// cycle nodes in input layer to calculate output of hidden layer
	for (unsigned node = 0; node < networkLayers[1].size(); node++) {
		networkLayers[1][node].gaussianRadialBasisFunction(inputLayer);
	}

	Layer hiddenLayer = networkLayers[1];
	// cycle nodes in hidden layer to calculate output of output layer
	for (unsigned node = 0; node < networkLayers[2].size(); node++) {
		networkLayers[2][node].calculateWeightedSum(hiddenLayer);
	}
}

// function used for training network
void Network::trainNetwork(std::vector<dataEntry> trainingData, std::vector<dataEntry> testingData) {
	// initialise variables needed during the training process
	int trainingNoMisclassified = 0, trainingNoEdibleMisclassified = 0, trainingNoPoisonousMisclassified = 0;
	int testingNoMisclassified = 0, testingNoEdibleMisclassified = 0, testingNoPoisonousMisclassified = 0;
	float trainingError = 0.0f, trainingTotalError = 0.0f;
	float testingError = 0.0f, testingTotalError = 0.0f;

	// becomes true when every training example output is within tolerance
	bool allWithinTolerance;
	// value of current epoch of training
	int currEpoch = 0;
	// assigns index of node that produces highest gaussian response
	int winningNodeIndex;
	// stores current highest gaussian response
	float highestVal = 0;

	// enter training loop
	while (currEpoch < maxEpochs) {
		// run all testing data before making changes to get current testing error rate
		for (unsigned i = 0; i < testingData.size(); i++) {
			// run testing entry through network
			dataEntry testingEntry = testingData[i];
			feedForward(testingEntry.features);
			// record error from testing entry
			testingError = std::abs(testingEntry.expectedClassification - networkLayers[2][0].getOutputVal());
			testingTotalError += std::abs(pow(testingEntry.expectedClassification - networkLayers[2][0].getOutputVal(), 2));
			// if entry was wrongly classified
			if (testingError >= 1) {
			testingNoMisclassified++;
				// if entry should have been classified as poisonous
				if (testingEntry.expectedClassification == -1) {
					testingNoPoisonousMisclassified++;
				}
				// if entry should have been classified as edible
				else if (testingEntry.expectedClassification == 1) {
					testingNoEdibleMisclassified++;
				}
			}
		}

		printf("epoch %d \n", currEpoch);

		// initialise as true at the start of each iteration
		allWithinTolerance = true;
		// loop all training examples
		for (unsigned i = 0; i < trainingData.size(); i++) {
			// assign current data entry
			dataEntry trainingEntry = trainingData[i];

			// feed single data entry through network
			feedForward(trainingEntry.features);

			// calculate output error for current training example
			trainingError = std::abs(trainingEntry.expectedClassification - networkLayers[2][0].getOutputVal());
			trainingTotalError += std::abs(pow(trainingEntry.expectedClassification - networkLayers[2][0].getOutputVal(), 2));

			if (trainingError > tolerance) {
				// record training data result before weight change to get current training error rate
				trainingNoMisclassified++;
				// if entry should have been classified as poisonous
				if (trainingEntry.expectedClassification == -1) {
					trainingNoPoisonousMisclassified++;
				}
				// if entry should have been classified as edible
				else if (trainingEntry.expectedClassification == 1) {
					trainingNoEdibleMisclassified++;
				}

				// cycle every node in hidden layer of network
				for (unsigned node = 0; node < networkLayers[1].size(); node++) {
					// check if current node has higher gaussian response as previous
					if (networkLayers[1][node].getOutputVal() > highestVal) {
						// if so assign new highest value
						highestVal = networkLayers[1][node].getOutputVal();
						// assign index of node to variable
						winningNodeIndex = node;
					} 
				}
				// alter weight in right direction
				networkLayers[1][winningNodeIndex].nodeOutputWeights[0].weight += learningRate * (trainingEntry.expectedClassification - networkLayers[2][0].getOutputVal());

				// flag that a training example was greater than error tolerance
				allWithinTolerance = false;
			} 
			// reset highest value and winning index for next data entry
			highestVal = 0.0f, winningNodeIndex = 0;
		}

		printf("\nTraining Set\n");
		// print current epoch MSE training error to console
		printResults(trainingNoMisclassified, trainingNoPoisonousMisclassified, trainingNoEdibleMisclassified);
		// reset variables for next pass
		trainingNoMisclassified = trainingNoEdibleMisclassified = trainingNoPoisonousMisclassified = 0;
		// calculate overall network average error
		trainingTotalError /= trainingData.size();
		printf("Training error after %d epochs = %f \n\n", currEpoch, trainingTotalError);

		printf("\nTesting Set\n");
		// print current epoch MSE testing error to console
		printResults(testingNoMisclassified, testingNoPoisonousMisclassified, testingNoEdibleMisclassified);
		// reset variables for next pass
		testingNoMisclassified = testingNoEdibleMisclassified = testingNoPoisonousMisclassified = 0;
		// calculate overall network average error
		testingTotalError /= testingData.size();
		printf("Testing error after %d epochs = %f \n\n", currEpoch, testingTotalError);
		printf("------------------------------------------------------------------------\n");

		// exit loop if all training example error are below tolerance
		if (allWithinTolerance == true) {
			printf("Training took %d epochs \n", currEpoch + 1);
			break;
		}

		// increment training epoch
		currEpoch++;
	}
}


int main() {

	// define variable required for program 
	dataReader d;
	clock_t beginClock;
	clock_t endClock;
	float ms;
	float error = 0.0f, totalError = 0.0f;
	int noOfMisclassified = 0, noOfEdibleMisclassified = 0, noOfPoisonousMisclassified = 0;
	bool nodeFound;
	int noOfNodesNotNeeded = -1; // start at -1 as network will start with 1 node 

	// start recording time
	beginClock = clock();
	d.loadDataFile("MushroomDataSetNormalisedTestMin - Copy.txt");

	// vectors to store training, testing and validation sets of data 
	std::vector<dataEntry> trainingSet = d.dataSet.trainingSet;
	std::vector<dataEntry> testingSet = d.dataSet.testingSet;
	std::vector<dataEntry> validationSet = d.dataSet.validationSet;

	// finish recording time and print to console
	endClock = clock();
	ms = 1000.0f * (endClock - beginClock) / CLOCKS_PER_SEC;
	printf("Loading and Processing Data took %fms \n", ms);

	// calculate number of features present in data set
	int noOfFeatures = d.dataSet.trainingSet[0].features.size();

	// define network topology
	int inputNodes = noOfFeatures;
	int hiddenNodes = 1;
	int outputNodes = 1;

	// initialise network topology
	std::vector<int> topology;
	topology.push_back(inputNodes);
	topology.push_back(hiddenNodes);
	topology.push_back(outputNodes);

	// create network
	printf("Initialising Network... \n");
	// start recording time
	beginClock = clock();
	Network myNetwork(topology, trainingSet);
	// finish recording time and print to console
	endClock = clock();
	ms = 1000.0f * (endClock - beginClock) / CLOCKS_PER_SEC;
	printf("Initialisation took %fms \n", ms);

	// recruit number of hidden nodes required to cover input space
	printf("Recruiting Hidden Nodes... \n");
	// start with 0 hidden nodes added
	int noOfHiddenNodes = 0;
	// start recording time
	beginClock = clock();
	for (unsigned i = 0; i < trainingSet.size(); i++) {
		dataEntry data = trainingSet[i];

		// feed training example through network
		myNetwork.feedForward(data.features);          
		nodeFound = false;

		// cycle all nodes in hidden layer
		for (unsigned j = 0; j < myNetwork.networkLayers[1].size(); j++) {
			// check if any nodes in hidden layer are giving a strong gaussian response
			if (myNetwork.networkLayers[1][j].getOutputVal() > 0.88) {
				nodeFound = true;
				// increment how offset actual node value is from i value
				noOfNodesNotNeeded++;
			}
		}

		// add new node if suitable node not found
		if (nodeFound == false) {
			// add new node into hidden layer
			myNetwork.networkLayers[1].push_back(Node(outputNodes, i - noOfNodesNotNeeded, trainingSet, 1));
			for (int k = 0; k < noOfFeatures; k++) {
				// add weights from input nodes to new hidden node
				myNetwork.networkLayers[0][k].nodeOutputWeights.push_back(Connection());
				// assign value of weight
				myNetwork.networkLayers[0][k].nodeOutputWeights.back().weight = data.features[k];
			};
			noOfHiddenNodes++;
		}
	}
	// finish recording time and print to console
	endClock = clock();
	ms = 1000.0f * (endClock - beginClock) / CLOCKS_PER_SEC;
	printf("Recruitment took %fms \n", ms);
	printf("Network has recruited %d hidden nodes \n", noOfHiddenNodes);

	// allows user to view initial details before program continues
	printf("\nPress enter to continue... \n");
	std::getchar();
	
	printf("Training Network... \n");

	// start recording time
	beginClock = clock();
	// train network
	myNetwork.trainNetwork(trainingSet, testingSet);
	// finish recording time and print to console
	endClock = clock();
	ms = 1000.0f * (endClock - beginClock) / CLOCKS_PER_SEC;

	printf("------------------------------------------------------------------------\n\n");
	printf("Training took %fms \n \n", ms);
	printf("Training Set: \n");
	for (unsigned i = 0; i < trainingSet.size(); i++) {
		dataEntry data = trainingSet[i];
		// feed training example through network
		myNetwork.feedForward(data.features);

		Layer outputLayer = myNetwork.networkLayers[2];
		// calculate error between expected and actual output values
		error = std::abs(data.expectedClassification - outputLayer[0].getOutputVal());
		// calculating total MSE error
		totalError += std::abs(pow(error, 2));

		// check if training entry was incorrectly classified
		if (error >= 1) {
			noOfMisclassified++;
			// if entry should have been classified as poisonous
			if (data.expectedClassification == -1) {
				noOfPoisonousMisclassified++;
				// if entry should have been classified as edible
			} else if (data.expectedClassification == 1) {
				noOfEdibleMisclassified++;
			}
		}
	}
	// calculate average MSE accross all training data
	totalError /= trainingSet.size();
	printf("MSE error across all training examples: %f \n \n", totalError);
	// print results to console
	printResults(noOfMisclassified, noOfPoisonousMisclassified, noOfEdibleMisclassified);

	// reset variables for testing data
	noOfMisclassified = noOfEdibleMisclassified = noOfPoisonousMisclassified = 0;
	totalError = 0.0;

	// run testing data through network
	printf("Testing Set: \n");
	for (unsigned i = 0; i < testingSet.size(); i++) {
		dataEntry data = testingSet[i];
		// feed training example through network
		myNetwork.feedForward(data.features);

		Layer outputLayer = myNetwork.networkLayers[2];
		// calculate error between expected and actual output values
		error = std::abs(data.expectedClassification - outputLayer[0].getOutputVal());
		totalError += std::abs(pow(error, 2));

		// check if training entry was incorrectly classified
		if (error >= 1) {
			noOfMisclassified++;
			// if entry should have been classified as poisonous
			if (data.expectedClassification == -1) {
				noOfPoisonousMisclassified++;
				// if entry should have been classified as edible
			} else if (data.expectedClassification == 1) {
				noOfEdibleMisclassified++;
			}
		}
	}
	// calculate average MSE accross all training data
	totalError /= testingSet.size();
	printf("MSE error across all test examples: %f \n \n", totalError);
	// print results to console
	printResults(noOfMisclassified, noOfPoisonousMisclassified, noOfEdibleMisclassified);

	// stop program from closing
	std::getchar();

}