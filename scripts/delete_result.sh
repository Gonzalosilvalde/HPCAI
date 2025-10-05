#!/bin/bash
source ./.env

cd $PROJECT_ROUTE

rm bert_test.err  bert_test.out
rm -rf bert_squad_model training_results
