#!/bin/bash

cd /home/joewandy/git/metabolomics_tools/justin/cv_results/beer3pos.is
./run.sh
./do_cv_local.sh

cd /home/joewandy/git/metabolomics_tools/justin/cv_results/urine37pos.is
./run.sh
./do_cv_local.sh

cd /home/joewandy/git/metabolomics_tools/justin/cv_results/beer3pos.is.3bags
./run.sh
./do_cv_local.sh

cd /home/joewandy/git/metabolomics_tools/justin/cv_results/urine37pos.is.3bags
./run.sh
./do_cv_local.sh
