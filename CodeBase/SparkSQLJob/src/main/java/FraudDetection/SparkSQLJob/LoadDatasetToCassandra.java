package FraudDetection.SparkSQLJob;

/**
 *@author apoorv, ekjot, hartaj, piyush 
 *
 */

import java.net.UnknownHostException;
import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;

import com.datastax.driver.core.Session;
import com.datastax.spark.connector.cql.CassandraConnector;

public class LoadDatasetToCassandra {

	final static Logger logger = Logger.getLogger(LoadDatasetToCassandra.class);

	public static void main(String[] args) {

		String mode = "cluster";
		String msg = "";

		if (args.length > 0 && args[0].equalsIgnoreCase("local"))
			mode = args[0];

		SparkSession sparkSession = getJavaSparkSessionWithCassandra(mode);

		if (sparkSession != null) {
			msg = "Logpoint 1: Successfully generated spark session.....";
			addLogMessage(msg, "I");
			prepareSchemaAndTableInCassandra(JavaSparkContext.fromSparkContext(sparkSession.sparkContext()));
			Dataset<Row> creditCardTransactionsData = getSparkDatasetFromCSV(sparkSession);
			long num_records = creditCardTransactionsData.count();

			if (creditCardTransactionsData != null && num_records > 0) {
				msg = "Logpoint 3: Successfully loaded:" + num_records + " records from CSV file.....";
				addLogMessage(msg, "I");
				saveTrainingDataToCassandara(creditCardTransactionsData);
			} else {
				msg = "Logpoint 3: Could not load data from CSV file.....";
				addLogMessage(msg, "E");
			}

		} else {
			msg = "Logpoint 1: Problem generating spark session.....";
			addLogMessage(msg, "E");
		}

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

	private static void prepareSchemaAndTableInCassandra(JavaSparkContext javaSparkContext) {
		String msg = "";
		CassandraConnector connector = CassandraConnector.apply(javaSparkContext.getConf());
		Session session = null;
		try {
			session = connector.openSession();
			session.execute("DROP KEYSPACE IF EXISTS fraud_detection");
			String query = "CREATE KEYSPACE fraud_detection WITH replication = {'class':'SimpleStrategy', 'replication_factor':1};";
			session.execute(query);
			session.execute("USE fraud_detection");
			session.execute("DROP TABLE IF EXISTS credit_transaction_training_data");
			query = "CREATE TABLE credit_transaction_training_data(" + "ID VARCHAR PRIMARY KEY, " + "TIME VARCHAR, "
					+ "V1 VARCHAR, " + "V2 VARCHAR, " + "V3 VARCHAR, " + "V4 VARCHAR, " + "V5 VARCHAR, "
					+ "V6 VARCHAR, " + "V7 VARCHAR, " + "V8 VARCHAR, " + "V9 VARCHAR, " + "V10 VARCHAR, "
					+ "V11 VARCHAR, " + "V12 VARCHAR, " + "V13 VARCHAR, " + "V14 VARCHAR, " + "V15 VARCHAR, "
					+ "V16 VARCHAR, " + "V17 VARCHAR, " + "V18 VARCHAR, " + "V19 VARCHAR, " + "V20 VARCHAR, "
					+ "V21 VARCHAR, " + "V22 VARCHAR, " + "V23 VARCHAR, " + "V24 VARCHAR, " + "V25 VARCHAR, "
					+ "V26 VARCHAR, " + "V27 VARCHAR, " + "V28 VARCHAR, " + "AMOUNT VARCHAR, " + "CLASS VARCHAR);";
			session.execute(query);
			msg = "Logpoint 2: Schema and table generated in Cassandra.....";
			addLogMessage(msg, "I");
		} catch (Exception e) {
			msg = "Logpoint 2: Schema and table generation failed in Cassandra.....";
			addLogMessage(msg, "E");
			addLogMessage(e.getMessage(), "E");
		} finally {
			session.close();
		}

	}

	private static void saveTrainingDataToCassandara(Dataset<Row> creditCardTransactionsData) {
		String msg = "";
		Map<String, String> keyspaceTableName = new HashMap<String, String>();
		try {
			keyspaceTableName.put("keyspace", "fraud_detection");
			keyspaceTableName.put("table", "credit_transaction_training_data");
			creditCardTransactionsData.write().format("org.apache.spark.sql.cassandra").options(keyspaceTableName)
			.mode(SaveMode.Append).save();
			msg = "Logpoint 4: Successfully saved data to Cassandra.....";
			addLogMessage(msg, "I");
		} catch (Exception e) {
			msg = "Logpoint 4: Saving data to Cassandra failed.....";
			addLogMessage(msg, "E");
			addLogMessage(e.getMessage(), "E");
		}

	}

	private static SparkSession getJavaSparkSessionWithCassandra(String mode) {
		SparkConf conf = null;
		String msg = "";
		if (mode.trim().equalsIgnoreCase("cluster")) {
			conf = new SparkConf().setAppName("Spark_Sql_Job");
			try {
				conf.set("spark.cassandra.connection.host", java.net.InetAddress.getLocalHost().getHostAddress());
			} catch (UnknownHostException e) {
				msg = "Cassndra host IP is not correct.....";
				addLogMessage(msg, "E");
				addLogMessage(e.getMessage(), "E");
			}
		} else {
			conf = new SparkConf().setAppName("Spark_Sql_Job").setMaster("local");
			try {
				conf.set("spark.cassandra.connection.host", java.net.InetAddress.getLocalHost().getHostAddress());
			} catch (UnknownHostException e) {
				System.out.println("Cassndra host IP is not correct.....");
				e.printStackTrace();
			}
		}

		conf.set("spark.cassandra.connection.port", "9042");
		conf.set("spark.cassandra.connection.timeout_ms", "5000");
		conf.set("spark.cassandra.read.timeout_ms", "200000");
		SparkSession sparkSession = SparkSession.builder().config(conf).getOrCreate();
		return sparkSession;
	}

	private static Dataset<Row> getSparkDatasetFromCSV(SparkSession sparkSession) {
		String path = System.getProperty("user.dir") + "\\src\\main\\java\\resources\\Transactions.csv";
		return sparkSession.read().format("csv").option("header", "true").load(path);
	}
}
