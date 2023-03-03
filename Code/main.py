import os
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, row_number

if os.path.exists('src.zip'):
    sys.path.insert(0, 'src.zip')
else:
    sys.path.insert(0, './Code/src')

from src.utilities import utils
class AccidentAnalysis:
    def __init__(self, path_to_config_file):
        input_file_paths = utils.read_yaml(path_to_config_file).get("INPUT_FILENAME")
        self.charges_df = utils.load_csv_data_to_df(spark, input_file_paths.get("Charges"))
        self.damages_df = utils.load_csv_data_to_df(spark, input_file_paths.get("Damages"))
        self.endorse_df = utils.load_csv_data_to_df(spark, input_file_paths.get("Endorse"))
        self.primary_person_df = utils.load_csv_data_to_df(spark, input_file_paths.get("Primary_Person"))
        self.units_df = utils.load_csv_data_to_df(spark, input_file_paths.get("Units"))
        self.restrict_df = utils.load_csv_data_to_df(spark, input_file_paths.get("Restrict"))

    def count_male_died_in_accidents(self, output_path, output_format):
        """
        Finds the crashes (accidents) in which number of persons killed are male
        :param output_path: output file path
        :param output_format: Write file format
        :return: dataframe count
        """
        df = self.primary_person_df.filter((self.primary_person_df["PRSN_GNDR_ID"] == "MALE") & \
                                           (self.primary_person_df["DEATH_CNT"] == 1)).select("CRASH_ID")
        utils.write_output(df, output_path, output_format)
        return df.count()

    def count_two_wheeler_accidents(self, output_path, output_format):
        """
        Finds the crashes where the vehicle type was 2 wheeler.
        :param output_format: Write file format
        :param output_path: output file path
        :return: dataframe count
        """
        df = self.units_df.filter(col("VEH_BODY_STYL_ID").like("%MOTORCYCLE%"))
        utils.write_output(df, output_path, output_format)

        return df.count()

    def state_with_highest_female_accident(self, output_path, output_format):
        """
        Finds state name with highest female accidents
        :param output_format: Write file format
        :param output_path: output file path
        :return: state name with highest female accidents
        """
        df = self.primary_person_df.select("CRASH_ID","DRVR_LIC_STATE_ID" ).filter(col("PRSN_GNDR_ID") == "FEMALE"). \
             groupBy("DRVR_LIC_STATE_ID").count().orderBy(col("count").desc())
        utils.write_output(df, output_path, output_format)

        return df.first()[0]

    def top_5_15_vehicle_contributing_to_injuries(self, output_path, output_format):
        """
        Finds Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
        :param output_format: Write file format
        :param output_path: output file path
        :return: Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
        """
        df_temp = self.units_df.filter(self.units_df.VEH_MAKE_ID != "NA"). \
            withColumn("casualities_count", self.units_df[35] + self.units_df[36]). \
            groupby("VEH_MAKE_ID").sum("casualities_count"). \
            withColumnRenamed("sum(casualities_count)", "total_casualties").orderBy(col("total_casualties").desc())
        df = df_temp.limit(15).subtract(df_temp.limit(5))
        utils.write_output(df, output_path, output_format)
        df.show(truncate=False)


    def top_ethnic_group_for_each_body_style(self, output_path, output_format):
        """
        Finds and show top ethnic user group of each unique body style that was involved in crashes
        :param output_format: Write file format
        :param output_path: output file path
        :return: None
        """
        values=["NA", "UNKNOWN", "NOT REPORTED", "OTHER  (EXPLAIN IN NARRATIVE)"]
        df_1 = self.units_df.select("VEH_BODY_STYL_ID","CRASH_ID").filter(~col("VEH_BODY_STYL_ID").isin(values))
        df_2 = self.primary_person_df.filter(~col("PRSN_ETHNICITY_ID").isin(["NA", "UNKNOWN"]))
        df_temp= df_1.join(df_2, on=['CRASH_ID'], how='inner' ).groupBy("VEH_BODY_STYL_ID","PRSN_ETHNICITY_ID").count()
        w = Window.partitionBy("VEH_BODY_STYL_ID" ).orderBy(col("count").desc())
        df= df_temp.withColumn("row",row_number().over(w)).filter(col("row")== 1).drop("count", "row")
        utils.write_output(df, output_path, output_format)
        df.show(truncate=False)

    def top_5_zip_codes_with_alcohols_as_contributing_factor(self, output_path, output_format):
        """
        Finds top 5 Zip Codes with the highest number crashes with alcohols as the contributing factor to a crash
        :param output_format: Write file format
        :param output_path: output file path
        :return: List of Zip Codes
        """
        values= ["PASSENGER CAR, 4-DOOR", "PASSENGER CAR, 2-DOOR"]
        df_temp = self.units_df.join(self.primary_person_df, on=['CRASH_ID'], how='inner'). \
            filter(col("CONTRIB_FACTR_1_ID").contains("ALCOHOL") | col("CONTRIB_FACTR_2_ID").like("%ALCOHOL%")). \
            filter(col("VEH_BODY_STYL_ID").isin(values)).groupBy("DRVR_ZIP").count().na.drop().orderBy(col("count").desc())
        df= df_temp.select("DRVR_ZIP").drop("count").limit(5)
        utils.write_output(df, output_path, output_format)
        df.show(truncate=False)


    def crash_ids_with_no_damage(self, output_path, output_format):
        """
        Counts Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4
        and car avails Insurance.
        :param output_format: Write file format
        :param output_path: output file path
        :return: List of crash ids
        """
        values=["NA", "NO DAMAGE", "INVALID VALUE"]
        df = self.damages_df.join(self.units_df, on=["CRASH_ID"], how='inner').filter(((col("VEH_DMAG_SCL_1_ID") > "DAMAGED 4")
            & (~col("VEH_DMAG_SCL_1_ID").isin(values))) | ((col("VEH_DMAG_SCL_2_ID") > "DAMAGED 4") & (~col("VEH_DMAG_SCL_2_ID").isin(values)))).filter(col("DAMAGED_PROPERTY") == "NONE"). \
            filter(col("FIN_RESP_TYPE_ID") == "PROOF OF LIABILITY INSURANCE").select("CRASH_ID").distinct()

        utils.write_output(df, output_path, output_format)
        df.show(truncate=False)

    def top_5_vehicle_brand(self, output_path, output_format):
        """
        Determines the Top 5 Vehicle Makes/Brands where drivers are charged with speeding related offences, has licensed
        Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of
        offences
        :param output_format: Write file format
        :param output_path: output file path
        :return List of Vehicle brands
        """
        value= ["NA", "UNLICENSED", "UNKNOWN"]
        na= ["NONE","NA"]
        top_10_used_vehicle_colors = self.units_df.filter(~col("VEH_COLOR_ID").isin(na)). \
            groupBy(col("VEH_COLOR_ID")).count().orderBy(col("count").desc()). \
            drop("count").limit(10).rdd.flatMap(lambda x: x).collect()
        top_25_state_list  = self.units_df.filter(  (~col("CONTRIB_FACTR_1_ID").isin(na)) |  (~col("CONTRIB_FACTR_2_ID").isin(na)) | (~col("CONTRIB_FACTR_P1_ID").isin(na))). \
            groupBy("VEH_LIC_STATE_ID").count().orderBy(col("count").desc()). \
            drop("count").limit(25).rdd.flatMap(lambda x: x).collect()

        df_temp= self.charges_df.join(self.primary_person_df, ["CRASH_ID","PRSN_NBR","UNIT_NBR"]).join(self.units_df,["CRASH_ID","UNIT_NBR"]). \
            select("VEH_COLOR_ID", "VEH_LIC_STATE_ID", "CRASH_ID", "UNIT_NBR", "VEH_MAKE_ID"). \
            filter(~col("DRVR_LIC_CLS_ID").isin(value)).filter(col("CHARGE").like("%SPEED%") )
        df= df_temp.filter((col("VEH_LIC_STATE_ID").isin(top_25_state_list)) & (col("VEH_COLOR_ID").isin(top_10_used_vehicle_colors))).groupBy("VEH_MAKE_ID").count().orderBy(col("count").desc()).limit(5).drop("count")
        utils.write_output(df, output_path, output_format)
        df.show(truncate=False)


if __name__ == '__main__':
    # Initialize sparks session
    spark = SparkSession \
        .builder \
        .appName("AccidentAnalysis") \
        .getOrCreate()

    config_file_path = "config.yaml"
    spark.sparkContext.setLogLevel("ERROR")

    acc = AccidentAnalysis(config_file_path)
    output_file_paths = utils.read_yaml(config_file_path).get("OUTPUT_PATH")
    file_format = utils.read_yaml(config_file_path).get("FILE_FORMAT")

    # 1. Find the number of crashes (accidents) in which number of persons killed are male?
    print("1. Result:")
    print(acc.count_male_died_in_accidents(output_file_paths.get(1), file_format.get("Output")))

    # 2. How many two-wheelers are booked for crashes?
    print("2. Result:")
    print(acc.count_two_wheeler_accidents(output_file_paths.get(2), file_format.get("Output")))

    # 3. Which state has the highest number of accidents in which females are involved?
    print("3. Result:")
    print(acc.state_with_highest_female_accident(output_file_paths.get(3),file_format.get("Output")))

    # 4. Which are the Top 5th to 15th VEH_MAKE_IDs that contribute to a largest number of injuries including death
    print("4. Result:")
    print(acc.top_5_15_vehicle_contributing_to_injuries(output_file_paths.get(4),file_format.get("Output")))

    # 5. For all the body styles involved in crashes, mention the top ethnic user group of each unique body style
    print("5. Result:")
    print(acc.top_ethnic_group_for_each_body_style(output_file_paths.get(5), file_format.get("Output")))

    # 6. Among the crashed cars, what are the Top 5 Zip Codes with the highest number crashes with alcohols as the
    # contributing factor to a crash (Use Driver Zip Code)
    print("6. Result:")
    print(acc.top_5_zip_codes_with_alcohols_as_contributing_factor(output_file_paths.get(6), file_format.get("Output")))

    # 7. Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4
    # and car avails Insurance
    print("7. Result:")
    print(acc.crash_ids_with_no_damage(output_file_paths.get(7), file_format.get("Output")))

    # 8. Determine the Top 5 Vehicle Makes/Brands where drivers are charged with speeding related offences, has licensed
    # Drivers, uses top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of
    # offences (to be deduced from the data)
    print("8. Result:")
    print(acc.top_5_vehicle_brand(output_file_paths.get(8), file_format.get("Output")))

    spark.stop()
