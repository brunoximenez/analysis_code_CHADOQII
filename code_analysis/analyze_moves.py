

'''

TO BE CODED !!
just a copy of what is in main code

'''


# # Analyze moves

# In[30]:


init_filled_traps = initial_bool
target_trap_list = mc.import_sel_rois()
trap_positions = mc.import_traps()
trap_nb = init_filled_traps.shape[1]
n_run = init_filled_traps.shape[0]
n_run_ps = assembled_bool_ps.shape[0]

# Computes the move for all the repetitions
moves = mc.compute_moves(init_filled_traps, target_trap_list, trap_positions)
moves_nb = []
print(len(moves))
for k in range(len(moves)):
    moves_nb.append(len(moves[k]))
print(np.mean(moves_nb)-14)


fail_transfer_from_trap, fail_move_proba, moves_per_trap, failed_dump_from_trap, dumps_per_trap, fail_dump_proba = compute_failed_transfer_proba(
    assembled_bool_ps, target_trap_list)


# In[ ]:


# The data needs to be reshaped and rotated before use
fail_transfer_proba_to_plot = fail_transfer_from_trap
transfer_proba_to_plot = 1 - \
    np.rot90(fail_transfer_proba_to_plot.reshape(size_array, size_array))
disable_cells = (moves_per_trap == 0.)
disable_cells = np.rot90(disable_cells.reshape(size_array, size_array))

# limits_1 = (0.45,0.55)
limits_1 = None

plot_heatmap(transfer_proba_to_plot, 1-fail_move_proba,
             main_res_str='Transfer proba per move',             is_disable_cell=disable_cells)
par_dir = os.path.dirname(os.getcwd())
base_name = os.path.basename(os.path.normpath(par_dir))
file_name = os.path.join(par_dir, base_name + '_transfer.png')
plt.savefig(file_name, bbox_inches='tight')

# The data needs to be reshaped and rotated before use
move_per_trap_to_plot = moves_per_trap
move_per_trap_to_plot = np.rot90(
    move_per_trap_to_plot.reshape(size_array, size_array))
move_per_trap_avg = np.mean(moves_per_trap)

# limits_1 = (0.45,0.55)
limits_1 = None

plot_heatmap(move_per_trap_to_plot, move_per_trap_avg,
             main_res_str='Avg number of transfer from each trap per run')
par_dir = os.path.dirname(os.getcwd())
base_name = os.path.basename(os.path.normpath(par_dir))
file_name = os.path.join(par_dir, base_name + '_moves.png')
plt.savefig(file_name, bbox_inches='tight')


# In[ ]:


# The data needs to be reshaped and rotated before use
fail_dump_proba_to_plot = failed_dump_from_trap
fail_dump_proba_to_plot = 1 - \
    np.rot90(fail_dump_proba_to_plot.reshape(size_array, size_array))
disable_cells = (dumps_per_trap == 0.)
disable_cells = np.rot90(disable_cells.reshape(size_array, size_array))

# limits_1 = (0.45,0.55)
limits_1 = None

plot_heatmap(fail_dump_proba_to_plot, 1-fail_dump_proba,
             main_res_str='Success proba per dumping',             is_disable_cell=disable_cells)
par_dir = os.path.dirname(os.getcwd())
base_name = os.path.basename(os.path.normpath(par_dir))
file_name = os.path.join(par_dir, base_name + '_dumping.png')
plt.savefig(file_name, bbox_inches='tight')

# The data needs to be reshaped and rotated before use
dump_per_trap_to_plot = dumps_per_trap
dump_per_trap_to_plot = np.rot90(
    dump_per_trap_to_plot.reshape(size_array, size_array))
dump_per_trap_avg = np.mean(dump_per_trap_to_plot)
disable_cells = np.zeros(trap_nb)
disable_cells[selectedROI_list] = 1.
disable_cells = (disable_cells == 1.)
disable_cells = np.rot90(disable_cells.reshape(size_array, size_array))

# limits_1 = (0.45,0.55)
limits_1 = None

plot_heatmap(dump_per_trap_to_plot, dump_per_trap_avg,
             main_res_str='Avg number of dumpings from each trap per run', is_disable_cell=disable_cells)
par_dir = os.path.dirname(os.getcwd())
base_name = os.path.basename(os.path.normpath(par_dir))
file_name = os.path.join(par_dir, base_name + '_dump_moves.png')
plt.savefig(file_name, bbox_inches='tight')
