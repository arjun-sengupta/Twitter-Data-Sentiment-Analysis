import os
import sys
os.chdir("/home/arjun/Final Year BTech/WorkSpace")
os.curdir

# Configure the environment. Set this up to the directory where
# Spark is installed
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/ProgramFiles/Spark/spark-1.6.0-bin-hadoop2.6'

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

# If Spark V1.4.x is detected, then add ' pyspark-shell' to
# the end of the 'PYSPARK_SUBMIT_ARGS' environment variable


pyspark_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
if not "pyspark-shell" in pyspark_submit_args: pyspark_submit_args += " pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args


#Add the following paths to the system path. Please check your installation
#to make sure that these zip files actually exist. The names might change
#as versions change.
sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","pyspark.zip"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","py4j-0.10.4-src.zip"))

#Initiate Spark context. Once this is done all other applications can run
from pyspark import SparkContext
from pyspark import SparkConf

# Optionally configure Spark Settings
conf=SparkConf()
conf.set("spark.executor.memory", "1g")
conf.set("spark.cores.max", "2")

conf.setAppName("Twitter Data Sentiment Analysis")

## Initialize SparkContext. Run only once. Otherwise you get multiple 
#Context Error.
sc = SparkContext('local[2]', conf=conf)
