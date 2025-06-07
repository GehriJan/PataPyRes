import os
import numpy as np
import math
from tqdm import tqdm

rel_distances = [2, 3, 5, 8, 13, 21, 33, 54]
relevance_graph_class_names = [
    "UniversalSetRelevanceGraph",
    "AdjacencySetRelevanceGraph",
    "MatrixRelevanceGraph",
]
if True:
    for relevance_graph_class_name in relevance_graph_class_names:
        # remove all files from class name specific run script directory
        approach_dir = f"{relevance_graph_class_name}"
        for f in os.listdir(approach_dir):
            os.remove(os.path.join(approach_dir, f))

        for rel_distance in rel_distances:
            file_name = f"{approach_dir}/starexec_run_PyRes_rd_{rel_distance:03d}"
            runscript_content = f"""
#!/bin/tcsh
#
# This is a StarExec runscript for PyRes. See https://www.starexec.org.
#
# To use this, build a StarExec package with "make starexec" and
# upload it to a StarExec cluster - then follow local instructions
#
#;-).
#source /etc/profile.d/modules.csh
setenv MODULEPATH /starexec/sw/modulefiles/Linux:/starexec/sw/modulefiles/Core:/starexec/sw/lmod/lmod/modulefiles/Core
module load python/3.13.0
python --version
which python3.13.0
echo -n "% Problem    : " ; head -2 $1 | tail -1 | sed -e "s/.*  : //"
set ProblemSPC=`grep " SPC " $1 | sed -e "s/.* : //"`
set default_flags="-tifbsVp -nlargest -HPickGiven5"
set rel_flags="-r {rel_distance:03d} -c {relevance_graph_class_name}"
set final=" "$1
set ecmd="python3 ./pyres-fof.py $default_flags $rel_flags $final"
python3 --version
if ( `expr "$ProblemSPC" : "FOF.*"` || `expr "$ProblemSPC" : "CNF.*"` ) then
    echo "% Command    : " $ecmd
    /home/starexec/bin/GetComputerInfo -p THIS Model CPUModel RAMPerCPU OS | \
        sed -e "s/Computer     /% Computer   /" \
            -e "s/Model        /% Model      /" \
            -e "s/CPUModel     /% CPU        /" \
            -e "s/RAMPerCPU    /% Memory     /" \
            -e "s/OS           /% OS         /"
    echo -n "% CPULimit   : " ; echo "$STAREXEC_CPU_LIMIT"
    echo -n "% DateTime   : " ; date
    echo "% CPUTime    : "
    $ecmd
else
    echo "% SZS status Inappropriate"
endif
"""

            with open(file_name, "w") as file:
                file.write(runscript_content.strip())
