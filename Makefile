# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

PYEXEC=python

remove=rm -rf
cl = clear


FILE1=rforest_pfracs.py
FILE2=predict_P.py
FILE3=table2raster.py
FILE4=var_importances.py
FILE5=di.py
FILE6=plot_maps.py
FILE7=P_quantiles.py
FILE8=partial_dependence_plots.py


pmaps: $(FILE1) $(FILE2) $(FILE3) $(FILE4) $(FILE5) $(FILE6) $(FILE7) $(FILE8)

#	$(PYEXEC) $(FILE1) occ_p
#	$(PYEXEC) $(FILE1) mineral_p
#	$(PYEXEC) $(FILE1) inorg_p
#	$(PYEXEC) $(FILE1) org_p
#	$(PYEXEC) $(FILE1) avail_p
#	$(PYEXEC) $(FILE1) total_p
#
#	$(PYEXEC) $(FILE2)
	$(PYEXEC) $(FILE3)
#	$(PYEXEC) $(FILE4)
#	$(PYEXEC) $(FILE5)
#	$(PYEXEC) $(FILE6)
#	$(PYEXEC) $(FILE7)
#	$(PYEXEC) $(FILE8)
#	$(PYEXEC) $(FILE9)

# clean_pkl:
# 	$(remove) *.pkl

# clean_nc4:
# 	$(remove) *.nc4

# clean_py:
# 	$(remove) __pycache__

# clean: clean_pkl clean_nc4 clean_py
# 	$(remove) p_figs
# 	$(remove) PDP_PLOTS
# 	$(remove) predicted_P_avail_p
# 	$(remove) predicted_P_inorg_p
# 	$(remove) predicted_P_org_p
# 	$(remove) predicted_P_total_p
# 	$(remove)  importances_* permutation_importances_* model_selection_scores.csv
# 	$(remove) eval_metrics* AOA_* DI_* HIST_*
# 	$(cl)
