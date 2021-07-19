import unittest
import math
import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from dymos.examples.hypersonic.hypersonic_ode import HypersonicODE

import numpy as np
import openmdao.api as om


@use_tempdirs
class TestHypersonic(unittest.TestCase):

    def test_hypersonic(self):
        import numpy as np
        import openmdao.api as om
        import dymos as dm
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=HypersonicODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10e4), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('h', fix_initial=True, fix_final=True, rate_source='hdot')
        phase.add_state('v', fix_initial=True, fix_final=False, rate_source='vdot')
        phase.add_state('theta', fix_initial=True, fix_final=True, rate_source='thetadot')
        phase.add_state('gamma', fix_initial=True, fix_final=False, rate_source='gammadot')
        phase.add_state('kinetic_energy', fix_initial=True, fix_final=False, rate_source='kinetic_energy_dot')

        # Define alpha as a control.
        phase.add_control(name='alpha', units='rad', lower=-math.pi/2, upper=math.pi/2)

        # Minimize final time.
        phase.add_objective('kinetic_energy', loc='final', scaler=-1)

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.
        p.set_val('traj.phase0.states:h', phase.interp('h', [80000, 0]), units='m')

        #phase.add_boundary_constraint('v', loc='initial', equals=4000)
        p.set_val('traj.phase0.states:v', phase.interp('v', [4000, 4000]), units='m/s')

        p.set_val('traj.phase0.states:theta', phase.interp('theta', [0, 1*math.pi/180]), units='rad')

        #phase.add_boundary_constraint('gamma', loc='initial', equals=-math.pi/4)
        p.set_val('traj.phase0.states:gamma', phase.interp('gamma', [-math.pi/4, -math.pi/4]), units='rad')

        #phase.add_boundary_constraint('kinetic_energy', loc='initial', equals=8000000)
        p.set_val('traj.phase0.states:kinetic_energy', phase.interp('kinetic_energy', [8000000, 8000000]), units='m**2/s**2')

        p.set_val('traj.phase0.controls:alpha', phase.interp('alpha', [-math.pi/2, math.pi/2]), units='rad')
        # Run the driver to solve the problem
        p.run_driver()

        # Check the validity of our results by using scipy.integrate.solve_ivp to
        # integrate the solution.
        sim_out = traj.simulate()

        # Plot the results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

        axes[0].plot(p.get_val('traj.phase0.timeseries.states:x'),
                     p.get_val('traj.phase0.timeseries.states:y'),
                     'ro', label='solution')

        axes[0].plot(sim_out.get_val('traj.phase0.timeseries.states:x'),
                     sim_out.get_val('traj.phase0.timeseries.states:y'),
                     'b-', label='simulation')

        axes[0].set_xlabel('x (m)')
        axes[0].set_ylabel('y (m/s)')
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
                     p.get_val('traj.phase0.timeseries.controls:theta', units='deg'),
                     'ro', label='solution')

        axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                     sim_out.get_val('traj.phase0.timeseries.controls:theta', units='deg'),
                     'b-', label='simulation')

        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel(r'$\theta$ (deg)')
        axes[1].legend()
        axes[1].grid()

        plt.show()
