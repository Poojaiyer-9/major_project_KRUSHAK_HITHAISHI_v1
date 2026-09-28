import React, { useState } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { Audio } from 'expo-av';
import SeverityBadge from '../components/SeverityBadge';

export default function ResultScreen({ route, navigation }) {
  const { result } = route.params || {};
  const [playingAudio, setPlayingAudio] = useState(false);
  const [soundObj, setSoundObj] = useState(null);

  const displayName = result?.disease_display || result?.disease_name || 'Unknown';
  const confidence  = result?.confidence != null ? (result.confidence * 100).toFixed(1) : '—';
  const protocol    = result?.treatment || {};
  const weather     = result?.weather || {};
  const medicineName = protocol.medicine_name;

  const playVoice = async () => {
    if (!result?.voice_file_url) return;
    try {
      setPlayingAudio(true);
      if (soundObj) { await soundObj.unloadAsync(); }
      const { sound } = await Audio.Sound.createAsync(
        { uri: result.voice_file_url },
        { shouldPlay: true },
      );
      setSoundObj(sound);
      sound.setOnPlaybackStatusUpdate(status => {
        if (status.didJustFinish) setPlayingAudio(false);
      });
    } catch {
      setPlayingAudio(false);
    }
  };

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.container}>

      {/* Demo banner */}
      {result?.demo && (
        <View style={styles.demoBanner}>
          <Text style={styles.demoText}>⚠️  Demo mode — no model loaded</Text>
        </View>
      )}

      {/* Disease header */}
      <View style={styles.card}>
        <Text style={styles.diseaseName}>{displayName}</Text>
        <View style={styles.metaRow}>
          <SeverityBadge severity={result?.severity || 'LOW'} />
          <View style={styles.confidenceBox}>
            <Text style={styles.confidenceLabel}>Confidence</Text>
            <Text style={styles.confidenceValue}>{confidence}%</Text>
          </View>
        </View>
      </View>

      {/* GradCAM heatmap */}
      {result?.heatmap_base64 ? (
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>📊 Disease Heatmap</Text>
          <Image
            source={{ uri: `data:image/jpeg;base64,${result.heatmap_base64}` }}
            style={styles.heatmap}
            resizeMode="cover"
          />
          <Text style={styles.heatmapCaption}>Highlighted regions show where the AI focused</Text>
        </View>
      ) : null}

      {/* Advisory */}
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>📋 Advisory</Text>
        <Text style={styles.advisoryText}>{result?.advisory_translated || result?.advisory_english || '—'}</Text>
        {result?.voice_file_url && (
          <TouchableOpacity style={styles.voiceButton} onPress={playVoice} disabled={playingAudio}>
            {playingAudio
              ? <ActivityIndicator color="#fff" size="small" />
              : <Text style={styles.voiceButtonText}>🔊  Play Voice Advisory</Text>
            }
          </TouchableOpacity>
        )}
      </View>

      {/* Treatment protocol */}
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>💊 Treatment Protocol</Text>
        <InfoRow icon="🧪" label="Medicine" value={protocol.medicine_name} />
        <InfoRow icon="⚖️"  label="Dosage" value={protocol.dosage_per_acre} />
        <InfoRow icon="🌿"  label="Organic Alt." value={protocol.organic_alternative} />
        <InfoRow icon="⏰"  label="Timing" value={protocol.application_timing} />
      </View>

      {/* Weather at time of scan */}
      {weather.temperature_2m != null && (
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>🌤️ Weather at Scan Location</Text>
          <View style={styles.weatherRow}>
            <WeatherStat icon="🌡️" label="Temp" value={`${weather.temperature_2m}°C`} />
            <WeatherStat icon="💧" label="Humidity" value={`${weather.relative_humidity_2m}%`} />
            <WeatherStat icon="🌧️" label="Rain" value={`${weather.precipitation} mm`} />
          </View>
        </View>
      )}

      {/* Actions */}
      <TouchableOpacity
        style={styles.shopsButton}
        onPress={() => navigation.navigate('Shops', { medicineName })}
      >
        <Text style={styles.shopsButtonText}>🏪  Find Nearby Shops for this Medicine</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.rescanButton}
        onPress={() => navigation.navigate('Scan')}
      >
        <Text style={styles.rescanButtonText}>📷  Scan Another Leaf</Text>
      </TouchableOpacity>

    </ScrollView>
  );
}

function InfoRow({ icon, label, value }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoIcon}>{icon}</Text>
      <View style={{ flex: 1 }}>
        <Text style={styles.infoLabel}>{label}</Text>
        <Text style={styles.infoValue}>{value || '—'}</Text>
      </View>
    </View>
  );
}

function WeatherStat({ icon, label, value }) {
  return (
    <View style={styles.weatherStat}>
      <Text style={styles.weatherIcon}>{icon}</Text>
      <Text style={styles.weatherValue}>{value}</Text>
      <Text style={styles.weatherLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#f0fdf4' },
  container: { padding: 16, gap: 12, paddingBottom: 32 },

  demoBanner: {
    backgroundColor: '#fef9c3',
    borderRadius: 10,
    padding: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#eab308',
  },
  demoText: { color: '#854d0e', fontWeight: '600', fontSize: 13 },

  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
    gap: 10,
  },

  diseaseName: { fontSize: 22, fontWeight: '800', color: '#111827' },
  metaRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },

  confidenceBox: {
    backgroundColor: '#f0fdf4',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 6,
    alignItems: 'center',
  },
  confidenceLabel: { fontSize: 11, color: '#6b7280', fontWeight: '600' },
  confidenceValue: { fontSize: 18, fontWeight: '800', color: '#16a34a' },

  sectionTitle: { fontSize: 15, fontWeight: '700', color: '#374151' },

  heatmap: { width: '100%', height: 220, borderRadius: 10 },
  heatmapCaption: { fontSize: 12, color: '#9ca3af', textAlign: 'center' },

  advisoryText: { fontSize: 14, color: '#374151', lineHeight: 22 },

  voiceButton: {
    backgroundColor: '#16a34a',
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 4,
  },
  voiceButtonText: { color: '#fff', fontWeight: '700', fontSize: 15 },

  infoRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  infoIcon: { fontSize: 18, marginTop: 2 },
  infoLabel: { fontSize: 12, color: '#6b7280', fontWeight: '600' },
  infoValue: { fontSize: 14, color: '#111827', marginTop: 2, lineHeight: 20 },

  weatherRow: { flexDirection: 'row', justifyContent: 'space-around' },
  weatherStat: { alignItems: 'center', gap: 4 },
  weatherIcon: { fontSize: 22 },
  weatherValue: { fontSize: 16, fontWeight: '700', color: '#111827' },
  weatherLabel: { fontSize: 11, color: '#9ca3af' },

  shopsButton: {
    backgroundColor: '#16a34a',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
    shadowColor: '#16a34a',
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 3,
  },
  shopsButtonText: { color: '#fff', fontWeight: '800', fontSize: 15 },

  rescanButton: {
    backgroundColor: '#fff',
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#16a34a',
  },
  rescanButtonText: { color: '#16a34a', fontWeight: '700', fontSize: 15 },
});
