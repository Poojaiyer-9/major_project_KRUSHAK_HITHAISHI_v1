import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  Image,
} from 'react-native';

export default function HomeScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#16a34a" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerEmoji}>🌾</Text>
        <Text style={styles.appName}>Krushak Hithaishi</Text>
        <Text style={styles.tagline}>ಕೃಷಕ ಹಿತೈಷಿ • Farmer's Friend</Text>
      </View>

      {/* Feature cards */}
      <View style={styles.features}>
        <View style={styles.featureRow}>
          <View style={styles.featureCard}>
            <Text style={styles.featureEmoji}>🔬</Text>
            <Text style={styles.featureTitle}>AI Detection</Text>
            <Text style={styles.featureDesc}>Instant leaf disease diagnosis</Text>
          </View>
          <View style={styles.featureCard}>
            <Text style={styles.featureEmoji}>🌤️</Text>
            <Text style={styles.featureTitle}>Weather</Text>
            <Text style={styles.featureDesc}>Live local conditions</Text>
          </View>
        </View>
        <View style={styles.featureRow}>
          <View style={styles.featureCard}>
            <Text style={styles.featureEmoji}>💊</Text>
            <Text style={styles.featureTitle}>Treatment</Text>
            <Text style={styles.featureDesc}>Medicine & organic options</Text>
          </View>
          <View style={styles.featureCard}>
            <Text style={styles.featureEmoji}>🗣️</Text>
            <Text style={styles.featureTitle}>Voice Advisory</Text>
            <Text style={styles.featureDesc}>Kannada / Hindi / Telugu</Text>
          </View>
        </View>
      </View>

      {/* CTA */}
      <TouchableOpacity
        style={styles.scanButton}
        onPress={() => navigation.navigate('Scan')}
        activeOpacity={0.85}
      >
        <Text style={styles.scanIcon}>📷</Text>
        <Text style={styles.scanButtonText}>Scan a Leaf</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.shopsButton}
        onPress={() => navigation.navigate('Shops', { medicineName: null })}
        activeOpacity={0.85}
      >
        <Text style={styles.shopsButtonText}>🏪  Find Nearby Agri Shops</Text>
      </TouchableOpacity>

      <Text style={styles.footer}>Supports Tomato • 10 disease classes</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f0fdf4' },

  header: {
    backgroundColor: '#16a34a',
    paddingTop: 56,
    paddingBottom: 32,
    alignItems: 'center',
  },
  headerEmoji: { fontSize: 48, marginBottom: 8 },
  appName: { fontSize: 26, fontWeight: '800', color: '#fff', letterSpacing: 0.5 },
  tagline: { fontSize: 13, color: '#bbf7d0', marginTop: 4 },

  features: { padding: 16, gap: 12 },
  featureRow: { flexDirection: 'row', gap: 12 },
  featureCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 14,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  featureEmoji: { fontSize: 28, marginBottom: 6 },
  featureTitle: { fontSize: 13, fontWeight: '700', color: '#111827', textAlign: 'center' },
  featureDesc: { fontSize: 11, color: '#6b7280', textAlign: 'center', marginTop: 2 },

  scanButton: {
    marginHorizontal: 20,
    marginTop: 8,
    backgroundColor: '#16a34a',
    borderRadius: 16,
    paddingVertical: 18,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    shadowColor: '#16a34a',
    shadowOpacity: 0.4,
    shadowRadius: 8,
    elevation: 4,
  },
  scanIcon: { fontSize: 22 },
  scanButtonText: { fontSize: 18, fontWeight: '800', color: '#fff' },

  shopsButton: {
    marginHorizontal: 20,
    marginTop: 12,
    backgroundColor: '#fff',
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#16a34a',
  },
  shopsButtonText: { fontSize: 15, fontWeight: '700', color: '#16a34a' },

  footer: { textAlign: 'center', color: '#9ca3af', fontSize: 12, marginTop: 16 },
});
