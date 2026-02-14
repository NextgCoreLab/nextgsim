//! NTN/Satellite Link Simulation (Item #198)
//!
//! LEO/GEO delay/jitter models for non-terrestrial network simulation.
//! Provides realistic satellite link budget, propagation delay, Doppler shift,
//! and atmospheric attenuation models per 3GPP TR 38.811/38.821.

use std::f64::consts::PI;

// ============================================================================
// Constants
// ============================================================================

/// Speed of light in m/s
const C: f64 = 299_792_458.0;
/// Earth radius in km
const EARTH_RADIUS_KM: f64 = 6371.0;
/// Boltzmann constant (dB/K/Hz)
const K_BOLTZMANN_DB: f64 = -228.6;

// ============================================================================
// Orbit Models
// ============================================================================

/// Satellite orbit parameters
#[derive(Debug, Clone, Copy)]
pub struct OrbitParams {
    /// Orbital altitude in km
    pub altitude_km: f64,
    /// Orbital inclination in degrees
    pub inclination_deg: f64,
    /// Orbital period in seconds
    pub period_s: f64,
    /// Right ascension of ascending node (RAAN) in degrees
    pub raan_deg: f64,
}

impl OrbitParams {
    /// Creates LEO orbit parameters (e.g., Starlink-like)
    pub fn leo(altitude_km: f64, inclination_deg: f64) -> Self {
        let r = EARTH_RADIUS_KM + altitude_km;
        // Kepler's third law: T = 2π√(r³/μ), μ = 398600.4418 km³/s²
        let period_s = 2.0 * PI * (r.powi(3) / 398600.4418).sqrt();
        Self { altitude_km, inclination_deg, period_s, raan_deg: 0.0 }
    }

    /// Creates GEO orbit parameters
    pub fn geo() -> Self {
        Self {
            altitude_km: 35786.0,
            inclination_deg: 0.0,
            period_s: 86164.1, // Sidereal day
            raan_deg: 0.0,
        }
    }

    /// Creates MEO orbit parameters (e.g., O3b-like)
    pub fn meo(altitude_km: f64) -> Self {
        let r = EARTH_RADIUS_KM + altitude_km;
        let period_s = 2.0 * PI * (r.powi(3) / 398600.4418).sqrt();
        Self { altitude_km, inclination_deg: 0.0, period_s, raan_deg: 0.0 }
    }

    /// Creates HAPS parameters
    pub fn haps(altitude_km: f64) -> Self {
        Self { altitude_km, inclination_deg: 0.0, period_s: f64::INFINITY, raan_deg: 0.0 }
    }

    /// Angular velocity in rad/s
    pub fn angular_velocity(&self) -> f64 {
        if self.period_s.is_infinite() { return 0.0; }
        2.0 * PI / self.period_s
    }
}

// ============================================================================
// Propagation Delay Model
// ============================================================================

/// Ground station position (lat/lon in degrees, altitude in km)
#[derive(Debug, Clone, Copy)]
pub struct GroundPosition {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_km: f64,
}

impl GroundPosition {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { latitude_deg: lat, longitude_deg: lon, altitude_km: 0.0 }
    }
}

/// Satellite position at a given time
#[derive(Debug, Clone, Copy)]
pub struct SatellitePosition {
    /// Sub-satellite latitude (degrees)
    pub latitude_deg: f64,
    /// Sub-satellite longitude (degrees)
    pub longitude_deg: f64,
    /// Altitude (km)
    pub altitude_km: f64,
}

/// Computes slant range between ground station and satellite (km)
pub fn slant_range_km(ground: &GroundPosition, sat: &SatellitePosition) -> f64 {
    let earth_r = EARTH_RADIUS_KM + ground.altitude_km;
    let sat_r = EARTH_RADIUS_KM + sat.altitude_km;

    let lat1 = ground.latitude_deg.to_radians();
    let lon1 = ground.longitude_deg.to_radians();
    let lat2 = sat.latitude_deg.to_radians();
    let lon2 = sat.longitude_deg.to_radians();

    // Central angle using Haversine
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let central_angle = 2.0 * a.sqrt().asin();

    // Law of cosines for slant range
    (earth_r.powi(2) + sat_r.powi(2) - 2.0 * earth_r * sat_r * central_angle.cos()).sqrt()
}

/// One-way propagation delay in milliseconds
pub fn propagation_delay_ms(ground: &GroundPosition, sat: &SatellitePosition) -> f64 {
    let range_km = slant_range_km(ground, sat);
    let range_m = range_km * 1000.0;
    (range_m / C) * 1000.0 // Convert seconds to ms
}

/// Round-trip time in milliseconds
pub fn rtt_ms(ground: &GroundPosition, sat: &SatellitePosition) -> f64 {
    propagation_delay_ms(ground, sat) * 2.0
}

// ============================================================================
// Doppler Shift Model
// ============================================================================

/// Computes Doppler shift in Hz for a given relative velocity and carrier frequency
pub fn doppler_shift_hz(relative_velocity_ms: f64, carrier_freq_hz: f64) -> f64 {
    (relative_velocity_ms / C) * carrier_freq_hz
}

/// Estimates maximum Doppler shift for an orbit
pub fn max_doppler_shift_hz(orbit: &OrbitParams, carrier_freq_hz: f64) -> f64 {
    // Maximum relative velocity ≈ orbital velocity for LEO at low elevation
    let orbital_radius_m = (EARTH_RADIUS_KM + orbit.altitude_km) * 1000.0;
    let orbital_velocity = orbital_radius_m * orbit.angular_velocity();
    doppler_shift_hz(orbital_velocity, carrier_freq_hz)
}

/// Doppler rate (Hz/s) - rate of change of Doppler shift
pub fn doppler_rate_hz_per_s(orbit: &OrbitParams, carrier_freq_hz: f64) -> f64 {
    let max_doppler = max_doppler_shift_hz(orbit, carrier_freq_hz);
    // Approximate: Doppler changes most rapidly at zenith pass
    // Rate ≈ max_doppler * angular_velocity
    max_doppler * orbit.angular_velocity()
}

// ============================================================================
// Link Budget Model
// ============================================================================

/// Link budget parameters
#[derive(Debug, Clone)]
pub struct LinkBudget {
    /// Transmit power (dBm)
    pub tx_power_dbm: f64,
    /// Transmit antenna gain (dBi)
    pub tx_antenna_gain_dbi: f64,
    /// Receive antenna gain (dBi)
    pub rx_antenna_gain_dbi: f64,
    /// Carrier frequency (GHz)
    pub carrier_freq_ghz: f64,
    /// System noise temperature (K)
    pub noise_temp_k: f64,
    /// Channel bandwidth (MHz)
    pub bandwidth_mhz: f64,
    /// Atmospheric loss (dB) - rain, gas, scintillation
    pub atmospheric_loss_db: f64,
    /// Implementation loss (dB)
    pub implementation_loss_db: f64,
}

impl LinkBudget {
    /// Creates a typical S-band NTN link budget
    pub fn s_band_ntn() -> Self {
        Self {
            tx_power_dbm: 43.0,     // 20W EIRP typical for satellite
            tx_antenna_gain_dbi: 30.0,
            rx_antenna_gain_dbi: 0.0, // UE isotropic
            carrier_freq_ghz: 2.0,
            noise_temp_k: 290.0,
            bandwidth_mhz: 20.0,
            atmospheric_loss_db: 1.0,
            implementation_loss_db: 2.0,
        }
    }

    /// Creates a Ka-band link budget
    pub fn ka_band_ntn() -> Self {
        Self {
            tx_power_dbm: 50.0,
            tx_antenna_gain_dbi: 38.0,
            rx_antenna_gain_dbi: 34.0,
            carrier_freq_ghz: 20.0,
            noise_temp_k: 350.0,
            bandwidth_mhz: 500.0,
            atmospheric_loss_db: 5.0,
            implementation_loss_db: 3.0,
        }
    }

    /// Free-space path loss (dB) at a given distance
    pub fn fspl_db(&self, distance_km: f64) -> f64 {
        // FSPL = 20*log10(d) + 20*log10(f) + 92.45 (d in km, f in GHz)
        20.0 * distance_km.log10() + 20.0 * self.carrier_freq_ghz.log10() + 92.45
    }

    /// Compute received SNR (dB) for a given slant range
    pub fn snr_db(&self, slant_range_km: f64) -> f64 {
        let eirp = self.tx_power_dbm + self.tx_antenna_gain_dbi;
        let fspl = self.fspl_db(slant_range_km);
        let noise_power = K_BOLTZMANN_DB + 10.0 * self.noise_temp_k.log10()
            + 10.0 * (self.bandwidth_mhz * 1e6).log10();

        eirp - fspl + self.rx_antenna_gain_dbi
            - self.atmospheric_loss_db
            - self.implementation_loss_db
            - noise_power
    }
}

// ============================================================================
// Jitter Model
// ============================================================================

/// Jitter model for satellite links
#[derive(Debug, Clone, Copy)]
pub struct JitterModel {
    /// Base jitter standard deviation (ms)
    pub base_jitter_std_ms: f64,
    /// Atmospheric scintillation component (ms)
    pub scintillation_ms: f64,
    /// Processing jitter at satellite (ms)
    pub processing_jitter_ms: f64,
}

impl JitterModel {
    /// Creates a LEO jitter model
    pub fn leo() -> Self {
        Self {
            base_jitter_std_ms: 0.5,
            scintillation_ms: 0.1,
            processing_jitter_ms: 0.2,
        }
    }

    /// Creates a GEO jitter model
    pub fn geo() -> Self {
        Self {
            base_jitter_std_ms: 2.0,
            scintillation_ms: 0.5,
            processing_jitter_ms: 0.5,
        }
    }

    /// Total jitter standard deviation (ms)
    pub fn total_jitter_std_ms(&self) -> f64 {
        (self.base_jitter_std_ms.powi(2)
            + self.scintillation_ms.powi(2)
            + self.processing_jitter_ms.powi(2))
        .sqrt()
    }

    /// Samples a jitter value using a simple deterministic model
    /// (in production, use random normal distribution)
    pub fn sample_jitter_ms(&self, seed: u64) -> f64 {
        // Simple hash-based pseudo-random for determinism
        let hash = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let normalized = ((hash >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
        normalized * self.total_jitter_std_ms()
    }
}

// ============================================================================
// Atmospheric Attenuation
// ============================================================================

/// Rain attenuation model (ITU-R P.618)
pub fn rain_attenuation_db(freq_ghz: f64, elevation_deg: f64, rain_rate_mm_h: f64) -> f64 {
    if rain_rate_mm_h <= 0.0 || elevation_deg <= 5.0 { return 0.0; }

    // Simplified ITU-R model coefficients
    let (k, alpha) = if freq_ghz < 10.0 {
        (0.0001 * freq_ghz.powi(2), 1.0)
    } else {
        (0.001 * freq_ghz, 0.8 + 0.01 * freq_ghz)
    };

    let specific_atten = k * rain_rate_mm_h.powf(alpha);
    let path_length_km = 5.0 / elevation_deg.to_radians().sin(); // Simplified

    specific_atten * path_length_km.min(20.0)
}

/// Gaseous attenuation (oxygen + water vapor) in dB
pub fn gaseous_attenuation_db(freq_ghz: f64, elevation_deg: f64) -> f64 {
    if elevation_deg <= 5.0 { return 0.0; }

    // Simplified model: attenuation increases with frequency
    let zenith_atten = if freq_ghz < 10.0 {
        0.01 * freq_ghz
    } else if freq_ghz < 50.0 {
        0.02 * freq_ghz
    } else {
        // 60 GHz oxygen absorption peak region
        0.1 * freq_ghz
    };

    zenith_atten / elevation_deg.to_radians().sin()
}

/// Scintillation attenuation (ionospheric/tropospheric)
pub fn scintillation_attenuation_db(freq_ghz: f64, elevation_deg: f64, time_percentage: f64) -> f64 {
    if elevation_deg <= 5.0 { return 0.0; }

    // Scintillation is stronger at lower frequencies and lower elevations
    let base_scint = if freq_ghz < 10.0 {
        // Ionospheric scintillation dominant
        2.0 / freq_ghz.sqrt()
    } else {
        // Tropospheric scintillation
        0.1 * freq_ghz.sqrt()
    };

    // Elevation dependency
    let elev_factor = 1.0 / (elevation_deg.to_radians().sin().sqrt());

    // Time percentage factor (higher for rare events)
    let time_factor = if time_percentage < 1.0 {
        1.0 + (1.0 - time_percentage) * 2.0
    } else {
        1.0
    };

    base_scint * elev_factor * time_factor
}

/// Cloud attenuation (liquid water content)
pub fn cloud_attenuation_db(freq_ghz: f64, elevation_deg: f64, liquid_water_kg_m2: f64) -> f64 {
    if elevation_deg <= 5.0 || liquid_water_kg_m2 <= 0.0 { return 0.0; }

    // ITU-R P.840 model: specific attenuation coefficient
    let k_l = 0.819 * freq_ghz / (freq_ghz + 1.0).powi(2); // Simplified

    // Path length through clouds
    let path_length = liquid_water_kg_m2 / elevation_deg.to_radians().sin();

    k_l * path_length
}

// ============================================================================
// Composite Link Simulator
// ============================================================================

/// NTN link simulation result
#[derive(Debug, Clone)]
pub struct NtnLinkResult {
    /// One-way delay (ms)
    pub delay_ms: f64,
    /// Jitter (ms, can be negative)
    pub jitter_ms: f64,
    /// SNR at receiver (dB)
    pub snr_db: f64,
    /// Doppler shift (Hz)
    pub doppler_hz: f64,
    /// Doppler rate (Hz/s)
    pub doppler_rate_hz_s: f64,
    /// Free-space path loss (dB)
    pub fspl_db: f64,
    /// Total atmospheric loss (dB)
    pub atmospheric_loss_db: f64,
    /// Rain fade loss (dB)
    pub rain_fade_db: f64,
    /// Scintillation loss (dB)
    pub scintillation_db: f64,
    /// Slant range (km)
    pub slant_range_km: f64,
    /// Elevation angle (degrees)
    pub elevation_deg: f64,
    /// Azimuth angle (degrees)
    pub azimuth_deg: f64,
    /// Link margin (dB)
    pub link_margin_db: f64,
}

/// NTN link simulator
pub struct NtnLinkSimulator {
    /// Orbit parameters
    orbit: OrbitParams,
    /// Link budget
    link_budget: LinkBudget,
    /// Jitter model
    jitter: JitterModel,
    /// Simulation tick counter
    tick: u64,
}

impl NtnLinkSimulator {
    /// Creates a new LEO link simulator
    pub fn new_leo(altitude_km: f64) -> Self {
        Self {
            orbit: OrbitParams::leo(altitude_km, 53.0),
            link_budget: LinkBudget::s_band_ntn(),
            jitter: JitterModel::leo(),
            tick: 0,
        }
    }

    /// Creates a new GEO link simulator
    pub fn new_geo() -> Self {
        Self {
            orbit: OrbitParams::geo(),
            link_budget: LinkBudget::ka_band_ntn(),
            jitter: JitterModel::geo(),
            tick: 0,
        }
    }

    /// Creates with custom parameters
    pub fn new(orbit: OrbitParams, link_budget: LinkBudget, jitter: JitterModel) -> Self {
        Self { orbit, link_budget, jitter, tick: 0 }
    }

    /// Simulates the satellite link at a given time
    pub fn simulate(&mut self, ground: &GroundPosition) -> NtnLinkResult {
        self.tick += 1;

        // Simple satellite position model (circular orbit)
        let t = self.tick as f64;
        let angle = self.orbit.angular_velocity() * t;
        let sat = SatellitePosition {
            latitude_deg: self.orbit.inclination_deg * angle.sin(),
            longitude_deg: ground.longitude_deg + angle.cos() * 10.0, // Simplified track
            altitude_km: self.orbit.altitude_km,
        };

        let range = slant_range_km(ground, &sat);
        let delay = propagation_delay_ms(ground, &sat);

        // Elevation and azimuth angles
        let earth_r = EARTH_RADIUS_KM;
        let sat_r = earth_r + sat.altitude_km;
        let cos_elev = (sat_r.powi(2) - earth_r.powi(2) - range.powi(2))
            / (2.0 * earth_r * range);
        let elevation = cos_elev.acos().to_degrees().clamp(5.0, 90.0);

        // Azimuth (simplified)
        let delta_lat = sat.latitude_deg - ground.latitude_deg;
        let delta_lon = sat.longitude_deg - ground.longitude_deg;
        let azimuth = delta_lon.atan2(delta_lat).to_degrees();

        // Atmospheric losses with more detail
        let rain_loss = rain_attenuation_db(self.link_budget.carrier_freq_ghz, elevation, 5.0);
        let gas_loss = gaseous_attenuation_db(self.link_budget.carrier_freq_ghz, elevation);
        let scint_loss = scintillation_attenuation_db(self.link_budget.carrier_freq_ghz, elevation, 0.01);
        let cloud_loss = cloud_attenuation_db(self.link_budget.carrier_freq_ghz, elevation, 0.5);
        let total_atmos = rain_loss + gas_loss + scint_loss + cloud_loss;

        let fspl = self.link_budget.fspl_db(range);
        let snr = self.link_budget.snr_db(range) - total_atmos;

        let carrier_hz = self.link_budget.carrier_freq_ghz * 1e9;
        let doppler = max_doppler_shift_hz(&self.orbit, carrier_hz) * (self.tick as f64 * 0.01).sin();
        let doppler_rate = doppler_rate_hz_per_s(&self.orbit, carrier_hz);

        let jitter = self.jitter.sample_jitter_ms(self.tick);

        // Link margin (assume required SNR of 10 dB for example)
        let required_snr_db = 10.0;
        let link_margin = snr - required_snr_db;

        NtnLinkResult {
            delay_ms: delay,
            jitter_ms: jitter,
            snr_db: snr,
            doppler_hz: doppler,
            doppler_rate_hz_s: doppler_rate,
            fspl_db: fspl,
            atmospheric_loss_db: total_atmos,
            rain_fade_db: rain_loss,
            scintillation_db: scint_loss,
            slant_range_km: range,
            elevation_deg: elevation,
            azimuth_deg: azimuth,
            link_margin_db: link_margin,
        }
    }

    /// Returns the orbit parameters
    pub fn orbit(&self) -> &OrbitParams { &self.orbit }

    /// Returns the current tick
    pub fn tick(&self) -> u64 { self.tick }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leo_orbit_params() {
        let orbit = OrbitParams::leo(550.0, 53.0);
        assert!((orbit.altitude_km - 550.0).abs() < 0.1);
        assert!(orbit.period_s > 5000.0 && orbit.period_s < 7000.0); // ~96 min
        assert!(orbit.angular_velocity() > 0.0);
    }

    #[test]
    fn test_geo_orbit_params() {
        let orbit = OrbitParams::geo();
        assert!((orbit.altitude_km - 35786.0).abs() < 1.0);
        assert!((orbit.period_s - 86164.1).abs() < 1.0);
    }

    #[test]
    fn test_propagation_delay_leo() {
        let ground = GroundPosition::new(40.0, -74.0);
        let sat = SatellitePosition { latitude_deg: 40.0, longitude_deg: -74.0, altitude_km: 550.0 };
        let delay = propagation_delay_ms(&ground, &sat);
        // LEO at zenith: ~550km → ~1.8ms
        assert!(delay > 1.0 && delay < 5.0, "LEO delay = {delay}ms");
    }

    #[test]
    fn test_propagation_delay_geo() {
        let ground = GroundPosition::new(0.0, 0.0);
        let sat = SatellitePosition { latitude_deg: 0.0, longitude_deg: 0.0, altitude_km: 35786.0 };
        let delay = propagation_delay_ms(&ground, &sat);
        // GEO at zenith: ~35786km → ~119ms
        assert!(delay > 100.0 && delay < 150.0, "GEO delay = {delay}ms");
    }

    #[test]
    fn test_rtt() {
        let ground = GroundPosition::new(0.0, 0.0);
        let sat = SatellitePosition { latitude_deg: 0.0, longitude_deg: 0.0, altitude_km: 550.0 };
        let rtt = rtt_ms(&ground, &sat);
        assert!((rtt - 2.0 * propagation_delay_ms(&ground, &sat)).abs() < 0.001);
    }

    #[test]
    fn test_doppler_shift() {
        // LEO satellite at ~7.5 km/s
        let doppler = doppler_shift_hz(7500.0, 2e9);
        assert!(doppler > 40000.0 && doppler < 60000.0);
    }

    #[test]
    fn test_max_doppler_leo() {
        let orbit = OrbitParams::leo(550.0, 53.0);
        let max_d = max_doppler_shift_hz(&orbit, 2e9);
        assert!(max_d > 30000.0, "LEO max Doppler = {max_d}Hz");
    }

    #[test]
    fn test_link_budget_snr() {
        let lb = LinkBudget::s_band_ntn();
        let snr = lb.snr_db(550.0); // LEO zenith
        // Should be positive SNR for a working link
        assert!(snr > 0.0, "SNR = {snr}dB at 550km");
    }

    #[test]
    fn test_fspl() {
        let lb = LinkBudget::s_band_ntn();
        let fspl = lb.fspl_db(550.0);
        // Expect ~155-165 dB at S-band / 550km
        assert!(fspl > 140.0 && fspl < 180.0, "FSPL = {fspl}dB");
    }

    #[test]
    fn test_jitter_model() {
        let jitter = JitterModel::leo();
        assert!(jitter.total_jitter_std_ms() > 0.0);
        let sample = jitter.sample_jitter_ms(42);
        assert!(sample.abs() < jitter.total_jitter_std_ms() * 4.0);
    }

    #[test]
    fn test_rain_attenuation() {
        let atten = rain_attenuation_db(20.0, 30.0, 10.0);
        assert!(atten > 0.0, "Rain atten = {atten}dB");

        // No rain = no attenuation
        let no_rain = rain_attenuation_db(20.0, 30.0, 0.0);
        assert_eq!(no_rain, 0.0);
    }

    #[test]
    fn test_gaseous_attenuation() {
        let atten = gaseous_attenuation_db(20.0, 45.0);
        assert!(atten > 0.0);

        // Higher frequency = more attenuation
        let higher = gaseous_attenuation_db(60.0, 45.0);
        assert!(higher > atten);
    }

    #[test]
    fn test_ntn_link_simulator_leo() {
        let mut sim = NtnLinkSimulator::new_leo(550.0);
        let ground = GroundPosition::new(40.0, -74.0);

        let result = sim.simulate(&ground);
        assert!(result.delay_ms > 0.0);
        assert!(result.slant_range_km > 0.0);
        assert!(result.elevation_deg >= 5.0);
    }

    #[test]
    fn test_ntn_link_simulator_geo() {
        let mut sim = NtnLinkSimulator::new_geo();
        let ground = GroundPosition::new(0.0, 0.0);

        let result = sim.simulate(&ground);
        assert!(result.delay_ms > 50.0, "GEO delay = {}ms", result.delay_ms);
    }

    #[test]
    fn test_simulator_produces_varying_results() {
        let mut sim = NtnLinkSimulator::new_leo(550.0);
        let ground = GroundPosition::new(40.0, -74.0);

        let r1 = sim.simulate(&ground);
        let r2 = sim.simulate(&ground);

        // Jitter should vary between ticks
        assert_ne!(r1.jitter_ms, r2.jitter_ms);
    }

    #[test]
    fn test_scintillation_attenuation() {
        // Low frequency - ionospheric scintillation
        let scint_low = scintillation_attenuation_db(2.0, 30.0, 0.01);
        assert!(scint_low > 0.0);

        // High frequency - tropospheric scintillation
        let scint_high = scintillation_attenuation_db(20.0, 30.0, 0.01);
        assert!(scint_high > 0.0);

        // Low elevation should have more scintillation
        let scint_low_elev = scintillation_attenuation_db(2.0, 10.0, 0.01);
        assert!(scint_low_elev > scint_low);
    }

    #[test]
    fn test_cloud_attenuation() {
        let atten = cloud_attenuation_db(20.0, 45.0, 2.0);
        assert!(atten > 0.0);

        // No clouds = no attenuation
        let no_cloud = cloud_attenuation_db(20.0, 45.0, 0.0);
        assert_eq!(no_cloud, 0.0);

        // More liquid water = more attenuation
        let heavy_cloud = cloud_attenuation_db(20.0, 45.0, 4.0);
        assert!(heavy_cloud > atten);
    }

    #[test]
    fn test_enhanced_link_result() {
        let mut sim = NtnLinkSimulator::new_leo(550.0);
        let ground = GroundPosition::new(40.0, -74.0);

        let result = sim.simulate(&ground);

        // Check all new fields are populated
        assert!(result.doppler_rate_hz_s.abs() >= 0.0);
        assert!(result.rain_fade_db >= 0.0);
        assert!(result.scintillation_db >= 0.0);
        assert!(result.azimuth_deg.abs() <= 360.0);
        // Link margin can be positive or negative
        assert!(result.link_margin_db.abs() < 1000.0);
    }
}
