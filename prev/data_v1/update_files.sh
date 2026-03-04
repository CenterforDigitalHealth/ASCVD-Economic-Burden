#!/bin/bash

SOURCE_PATH="../dataPreparation"
TARGET_PATH="."
# alpha.csv
# gd.csv
# dl1_countrycodeorg_country_name.csv
# tmp.ipynb

##############################
cp  ${SOURCE_PATH}/education/education_filled.csv ${TARGET_PATH}/education_filled.csv
cp  ${SOURCE_PATH}/GDP/GDP_ppp.csv ${TARGET_PATH}/GDP_ppp.csv
cp  ${SOURCE_PATH}/GDP/GDP_ppp_cia.csv ${TARGET_PATH}/GDP_ppp_cia.csv
cp  ${SOURCE_PATH}/health_expenditure/hepc_ppp.csv ${TARGET_PATH}/hepc_ppp.csv
cp  ${SOURCE_PATH}/labor/laborparticipation_final.csv ${TARGET_PATH}/laborparticipation_final.csv
cp  ${SOURCE_PATH}/physicalcapital/physical_ppp.csv ${TARGET_PATH}/physical_ppp.csv
cp  ${SOURCE_PATH}/population/population_un.csv ${TARGET_PATH}/population_un.csv
cp  ${SOURCE_PATH}/population/population_total.csv ${TARGET_PATH}/population_total.csv
cp  ${SOURCE_PATH}/savings/savings.csv ${TARGET_PATH}/savings.csv
cp  ${SOURCE_PATH}/TreatmentCost/TC_ppp.csv ${TARGET_PATH}/TC_ppp.csv
cp  ${SOURCE_PATH}/TreatmentCost/adjust_prevalence/prevalence.csv ${TARGET_PATH}/prevalence.csv
################################################################







