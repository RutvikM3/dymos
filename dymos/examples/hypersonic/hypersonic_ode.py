import numpy as np
import openmdao.api as om
import math


class HypersonicODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('max_q_dot', types=int,
                             desc='Maximum Allowed heat rate')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('h', val=np.zeros(nn), desc='height', units='m')

        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('theta', val=np.zeros(nn), desc='downrange angle', units='rad')

        self.add_input('gamma', val=np.zeros(nn), desc='flight path angle', units='rad')

        self.add_input('alpha', val=np.zeros(nn), desc='angle of attack', units='rad')

        # specific kinetic energy
        self.add_input('kinetic_energy', val=np.zeros(nn), desc = 'specific KE', units='m**2/s**2')

        # Constants
        self.add_input('R_e', val=6.378e6, desc='radius of earth', units='m')
        self.add_input('rho_s', val=1.2, desc='reference density', units='kg/m**3')
        self.add_input('h_s', val=7500, desc='reference height', units='m')
        self.add_input('C_l1', val=1)
        self.add_input('C_l0', val=1)
        self.add_input('C_d2', val=1)
        self.add_input('C_d1', val=1)
        self.add_input('C_d0', val=1)
        self.add_input('A_ref', val=1, desc='reference area', units='m**2')
        self.add_input('m', val=1, desc='mass', units='kg')

        self.add_input('mu', val=3.986e14, desc='gravitational parameter', units='m**3/s**2')
        self.add_output('hdot', val=np.zeros(nn), desc='velocity magnitude', units='m/s',
                        tags=['dymos.state_rate_source:h', 'dymos.state_units:m'])

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])

        self.add_output('thetadot', val=np.zeros(nn), desc='angle rate', units='rad/s',
                        tags=['dymos.state_rate_source:theta', 'dymos.state_units:rad'])

        self.add_output('gammadot', val=np.zeros(nn), desc='flight path angle rate', units='rad/s',
                        tags=['dymos.state_rate_source:gamma', 'dymos.state_units:rad'])

        self.add_output('kinetic_energy_dot', val=np.zeros(nn), desc='specific kinetic energy rate', units='m**2/s**3',
                        tags=['dymos.state_rate_source:kinetic_energy', 'dymos.state_units:m**2/s**2'])
        # self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
        #               units='m/s')

        # Setup partials
        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        h = inputs['h']
        v = inputs['v']
        theta = inputs['theta']
        gamma = inputs['gamma']
        alpha = inputs['alpha']
        ke = inputs['kinetic_energy']

        r = inputs['R_e'] + h
        rho = inputs['rho_s']*math.e**(-h/inputs['h_s'])
        q = 0.5*rho*v**2
        C_l = inputs["C_l1"]*alpha + inputs["C_l0"]
        C_d = inputs["C_d2"]*alpha**2 + inputs["C_d1"]*alpha + inputs["C_d0"]
        D = q*C_d*inputs['A_ref']
        L = q*C_l*inputs['A_ref']

        outputs['hdot'] = v*np.sin(gamma)
        vdot = -D/inputs['m'] - inputs['mu']*np.sin(gamma)/r**2
        outputs['vdot'] = vdot
        outputs['thetadot'] = v*np.cos(gamma)/r
        outputs['gammadot'] = L/inputs['m']/v + (v/r-inputs['mu']/v/r**2)*np.cos(gamma)
        outputs['kinetic_energy_dot'] = v*vdot
        #

    # def compute_partials(self, inputs, partials):

