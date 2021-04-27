package FraudDetection.SparkMLJob;

/**
 *@author apoorv, ekjot, hartaj, piyush 
 *
 */

import java.net.UnknownHostException;
import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import com.datastax.spark.connector.japi.CassandraJavaUtil;
import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;

import scala.Tuple2;

public class TrainModel {

	final static Logger logger = Logger.getLogger(TrainModel.class);

	public static void main(String[] args) {
		String mode = "cluster";
		String msg = "";

		JavaSparkContext javaSparkContext = null;
		JavaRDD<MyRow> dataset = null;
		JavaRDD<LabeledPoint> trainingData = null;
		JavaRDD<LabeledPoint> testData = null;
		String keySpaceName = "fraud_detection";
		String tableName = "credit_transaction_training_data";

		RandomForestModel model = null;

		if (args.length > 0 && args[0].equalsIgnoreCase("local"))
			mode = args[0];

		try {
			javaSparkContext = setJavaSparkContextWithCassandra(mode);

			if (javaSparkContext != null) {

				msg = "Logpoint 1: Successfully generated spark context.....";
				addLogMessage(msg, "I");

				dataset = readFromCassandra(javaSparkContext, tableName, keySpaceName);

				if (dataset != null) {
					long num_records = dataset.count();
					msg = "Logpoint 2: Successfully loaded:" + num_records + " records from Cassandra.....";
					addLogMessage(msg, "I");
					trainingData = dataset.map((MyRow myRow) -> {
						return new LabeledPoint(myRow.getClas(), myRow.getRowVector());
					});

					//JavaRDD<LabeledPoint>[] splits = splitDataSet(dataset);
					//trainingData = splits[0];
					//testData = splits[1];

					model = trainModel(trainingData);
					if (model != null) {
						//testModel(testData, model);
						saveModel(model, javaSparkContext);
					} else {
						msg = "Logpoint 3: Model could not be trained.....";
						addLogMessage(msg, "E");
					}
				} else {
					msg = "Logpoint 2: Could not load data from Cassandra.....";
					addLogMessage(msg, "E");
				}
			} else {
				msg = "Logpoint 1: Problem generating spark context.....";
				addLogMessage(msg, "E");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void saveModel(RandomForestModel model, JavaSparkContext javaSparkContext) {
		String path = System.getProperty("user.dir") + "\\model\\" + "\\RandomForestClassificationModel\\";
		String msg = "";
		try {
			model.save(javaSparkContext.sc(), path);
			msg = "Logpoint 5: Successfully saved model to: " + path;
			addLogMessage(msg, "I");
		} catch (Exception e) {
			msg = "Logpoint 5: Model could not be saved to local storage.....";
			addLogMessage(msg, "E");
			addLogMessage(e.getMessage(), "E");
		}
	}

	private static void testModel(JavaRDD<LabeledPoint> testData, RandomForestModel model) {
		// Evaluate model on test instances and compute test error

		JavaPairRDD<Double, Double> predictionAndLabel = testData
				.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		double testErr = predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testData.count();

		double accuracy = 100 - (testErr * 100);
		String msg = "Logpoint 4: Calculated test set accuracy of Random Forest model to be " + accuracy + "%.";
		addLogMessage(msg, "I");
	}

	private static RandomForestModel trainModel(JavaRDD<LabeledPoint> trainingData) {
		// Train a RandomForest model.
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		int numClasses = 2;
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		int numTrees = 300;
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini";
		int maxDepth = 5;
		int maxBins = 32;
		int seed = 12345;
		String msg = "";
		try {
			RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
					numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
			msg = "Logpoint 3: Successfully trained a Random Forest model on the dataset.....";
			addLogMessage(msg, "I");
			return model;
		} catch (Exception e) {
			msg = "Logpoint 3: Training a Random Forest model on the dataset failed.....";
			addLogMessage(msg, "E");
			addLogMessage(e.getMessage(), "E");
		}
		return null;
	}

	private static JavaRDD<LabeledPoint>[] splitDataSet(JavaRDD<MyRow> dataset) {
		JavaRDD<LabeledPoint> labeledDataSet = dataset.map((MyRow myRow) -> {
			return new LabeledPoint(myRow.getClas(), myRow.getRowVector());
		});

		JavaRDD<LabeledPoint>[] splits = labeledDataSet.randomSplit(new double[] { 0.8, 0.2 }, 11L);
		String msg = "Logpoint 3: Successfully created training and test data.....";
		addLogMessage(msg, "I");

		return splits;

	}

	private static JavaRDD<MyRow> readFromCassandra(JavaSparkContext javaSparkContext, String tableName,
			String keySpaceName) {

		CassandraJavaRDD<CassandraRow> cassandraRDD = CassandraJavaUtil.javaFunctions(javaSparkContext)
				.cassandraTable(keySpaceName, tableName);

		return cassandraRDD.map(new Function<CassandraRow, MyRow>() {
			private static final long serialVersionUID = 1L;

			@Override
			public MyRow call(CassandraRow row) throws Exception {
				double[] v = new double[28];
				for (int i = 0; i < 28; i++) {
					v[i] = Double.parseDouble(row.getString("v" + (i + 1)).trim());
				}
				MyRow myRow = new MyRow();
				myRow.setTime(Double.parseDouble(row.getString("time").trim()));
				myRow.setClas(Double.parseDouble(row.getString("class").trim()));
				myRow.setAmount(Double.parseDouble(row.getString("amount").trim()));
				myRow.setV(v);
				return myRow;
			}
		});
	}

	private static JavaSparkContext setJavaSparkContextWithCassandra(String mode) {

		SparkConf sparkConf = new SparkConf();

		sparkConf.setAppName("Spark_ML_Job");

		if (mode.equals("local")) {
			sparkConf.setMaster("local");
			try {
				sparkConf.set("spark.cassandra.connection.host", java.net.InetAddress.getLocalHost().getHostAddress());
			} catch (UnknownHostException e) {
				System.out.println("Cassndra host IP is not correct.....");
				e.printStackTrace();
			}
		} else {
			try {
				sparkConf.set("spark.cassandra.connection.host", java.net.InetAddress.getLocalHost().getHostAddress());
			} catch (UnknownHostException e) {
				System.out.println("Cassndra host IP is not correct.....");
				e.printStackTrace();
			}
		}
		sparkConf.set("spark.cassandra.connection.port", "9042");
		sparkConf.set("spark.cassandra.connection.timeout_ms", "5000");
		sparkConf.set("spark.cassandra.read.timeout_ms", "200000");
		sparkConf.set("spark.cassandra.auth.username", "test_user");
		sparkConf.set("spark.cassandra.auth.password", "test_password");
		sparkConf.set("spark.testing.memory", "2147480000");
		// sparkConf.set("spark.driver.memory", "2g");
		return new JavaSparkContext(sparkConf);
	}

	private static void addLogMessage(String msg, String type) {
		if (logger.isDebugEnabled()) {
			logger.debug(msg);
		}
		if (type.equalsIgnoreCase("E")) {
			logger.error(msg);
		} else {
			logger.info(msg);
		}
	}
}
