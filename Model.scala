
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import spark.implicits._

var supp = 0.07
var conf = 0.7
var flag = false

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").format("csv").load("BigData/dfnew.csv/*.csv")

val Array(training, testing) = df.randomSplit(Array(0.8, 0.2))
//////// Selecting data as one column Array ///////

val moData = training.select(array(df.columns.map(df(_)) : _*) as "items")

////////// Model implementation and fitting ////////

val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(supp).setMinConfidence(conf)
val model = fpgrowth.fit(moData)

//Filtering columns Month and Season sololy

val moFilArr  = df.select($"Month" ).distinct.collect().map(_.getString(0))
val seaFilArr = df.select($"Season").distinct.collect().map(_.getString(0))
val filArr = moFilArr ++ seaFilArr

// Display frequent itemsets.


var fi = model.freqItemsets.withColumn("items", concat_ws(" : " , $"items"))


fi = fi.withColumn("freq", $"freq" / df.select(count("Month")).first() )

fi = fi.filter("items not like 'NA %' and items not like '%NA %' ")

for (x <- filArr){
  fi = fi.filter($"items" !== x)
}

(fi.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true")
           .save("C://spark/My Programs/Project/output/temp/frequentItemsets.csv"))



// Display generated association rules.

var ar = (model.associationRules.withColumn("antecedent", concat_ws(" : " , $"antecedent"))
                                .withColumn("consequent", concat_ws(" : " , $"consequent")))

ar = (ar.filter(  "consequent not like 'NA %' and consequent not like '%NA %' ")
        .filter(  "antecedent not like 'NA %' and antecedent not like '%NA %' "))

for (x <- filArr){
  ar = ar.filter($"consequent" !== x)
}


(ar.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true")
           .save("C://spark/My Programs/Project/output/temp/associationRules.csv"))


// transform examines the input items against all the association rules and summarize the
// consequents as prediction

// The user input dataframe has the testing set default value
/*
if (flag == true){
  val dftr = spark.read.option("header","true").option("inferSchema","true").format("csv").load("input/data/*.csv")
  dftr = formalize(dftr)
}
*/
*/

var datatr = testing
for(x <- df.columns){
  datatr = datatr.withColumn(x ,when(col(x).contains("NA "), lit(null)).otherwise(col(x)))
}

datatr = datatr.limit(50)select(array(datatr.columns.map(datatr(_)) : _*) as "items")


var transform = (model.transform(datatr).withColumn("items", concat_ws(" : " , $"items"))
                                        .withColumn("prediction", concat_ws(" " , $"prediction")))

val win = Window.orderBy($"items")
transform = transform.withColumn("Row", row_number().over(win))

transform = (transform.withColumn("prediction", regexp_replace(col("prediction"), "(NA )[a-zA-Z]+([0-9]|\\~[3])*( )*", ""))
                           .filter(col("prediction") !== "")
                           .select($"Row", $"items",$"prediction"))

(transform.write.format("com.databricks.spark.csv").option("header", "true")
                       .save("C://spark/My Programs/Project/output/temp/transform.csv"))


//.withColumn("prediction", concat_ws(" : " , $"prediction")))
//.filter(  "prediction not like '' and prediction not like 'NA %' and prediction not like '%NA %'")
//map(x => x.filter(! _.contains("NA ")))
/////////
