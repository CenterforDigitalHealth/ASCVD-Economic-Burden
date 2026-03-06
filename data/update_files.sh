#!/bin/bash

SOURCE_PATH="/Users/caozhong/HMM_dataPreparation"
TARGET_PATH="."

##############################
cp  ${SOURCE_PATH}/alphabeta/alpha.csv ${TARGET_PATH}/
cp  ${SOURCE_PATH}/alphabeta/dl1_countrycodeorg_country_name.csv ${TARGET_PATH}/
cp  ${SOURCE_PATH}/alphabeta/gd.csv ${TARGET_PATH}/

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
cp  ${SOURCE_PATH}/population/population_gbd.csv ${TARGET_PATH}/population_gbd.csv
cp  ${SOURCE_PATH}/savings/savings.csv ${TARGET_PATH}/savings.csv
################################################################







