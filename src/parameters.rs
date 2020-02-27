use {serde::Deserialize, std::f64::consts::PI};

/// Simulation parameters
#[derive(Debug, PartialEq, Default, Deserialize)]
pub struct Parameters {
    pub numerical: Numerical,
    pub physical: Physical,
}

#[derive(Debug, PartialEq, Deserialize)]
pub struct Numerical {
    /// Inversion grid resolution in both x and y
    pub grid_resolution: usize,
    /// Number of vertical layers
    pub vertical_layers: usize,
    /// Simulation time step size
    pub time_step: f64,
    /// Total duration of the simulation
    pub duration: f64,
    /// Interval of saving grid data
    pub save_interval: f64,
    /// Maximum pressure difference on convergence
    pub max_pressure_difference: f64,
    /// Strip width
    pub strip_width: f64,
    /// Value of A2 in displacement equation
    pub a2: f64,
    /// Value of A3 in displacement equation
    pub a3: f64,
}
impl Default for Numerical {
    fn default() -> Self {
        Numerical {
            grid_resolution: 32,
            vertical_layers: 4,
            time_step: 1.0 / 32.0,
            duration: 25.0,
            save_interval: 0.25,
            max_pressure_difference: 1.0e-9,
            strip_width: 0.4,
            a2: 0.02,
            a3: -0.01,
        }
    }
}

#[derive(Debug, PartialEq, Deserialize)]
pub struct Physical {
    /// Constant Coriolis frequency f
    pub coriolis_frequency: f64,
    /// Short-scale gravity wave speed c
    pub gravity_wave_speed: f64,
    /// Mean fluid depth (conserved by mass conservation)
    pub mean_fluid_depth: f64,
    /// This times f is the damping rate on wavenumber grid_resolution/2
    pub damping: f64,
    /// Viscosity constant
    pub nnu: f64,
}

impl Default for Physical {
    fn default() -> Self {
        Physical {
            coriolis_frequency: 4.0 * PI,
            gravity_wave_speed: 2.0 * PI,
            mean_fluid_depth: 0.4,
            damping: 10.0,
            nnu: 3.0,
        }
    }
}

#[cfg(test)]
mod test {
    use {super::*, std::fs::File};

    #[test]
    fn defaults() {
        assert_eq!(
            Parameters::default(),
            serde_yaml::from_reader::<_, Parameters>(
                File::open("src/testdata/defaults.yaml").unwrap()
            )
            .unwrap()
        );
    }
}
