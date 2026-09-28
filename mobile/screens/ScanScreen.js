import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Dimensions,
  ScrollView,
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Location from 'expo-location';
import { detectDisease } from '../services/api';

const CROP_STAGES = [
  { label: '🌱  Seedling', value: 'seedling' },
  { label: '🌿  Vegetative', value: 'vegetative' },
  { label: '🌸  Flowering', value: 'flowering' },
];

const LANGUAGES = [
  { label: '🇮🇳  Kannada', value: 'kn' },
  { label: '🇮🇳  Hindi', value: 'hi' },
  { label: '🇮🇳  Telugu', value: 'te' },
  { label: '🇬🇧  English', value: 'en' },
];

const { width } = Dimensions.get('window');

export default function ScanScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [locationOk, setLocationOk] = useState(false);
  const [loading, setLoading] = useState(false);
  const [cropStage, setCropStage] = useState('vegetative');
  const [language, setLanguage] = useState('kn');
  const [statusMsg, setStatusMsg] = useState('');
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      if (!permission?.granted) await requestPermission();
      const loc = await Location.requestForegroundPermissionsAsync();
      setLocationOk(loc.status === 'granted');
    })();
  }, []);

  const handleCapture = async () => {
    if (!cameraRef.current) return;
    try {
      setLoading(true);
      setStatusMsg('📍 Getting location...');
      const location = await Location.getCurrentPositionAsync({});

      setStatusMsg('📷 Capturing photo...');
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });

      setStatusMsg('🔬 Analysing leaf...');
      const result = await detectDisease(
        photo.uri,
        location.coords.latitude,
        location.coords.longitude,
        cropStage,
        language,
      );
      navigation.navigate('Result', { result });
    } catch (err) {
      setStatusMsg('❌ Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!permission) return <ActivityIndicator style={{ flex: 1 }} />;

  if (!permission.granted) {
    return (
      <View style={styles.permissionBox}>
        <Text style={styles.permissionEmoji}>📷</Text>
        <Text style={styles.permissionTitle}>Camera Permission Needed</Text>
        <Text style={styles.permissionDesc}>
          Krushak Hithaishi needs camera access to photograph the leaf.
        </Text>
        <TouchableOpacity style={styles.grantButton} onPress={requestPermission}>
          <Text style={styles.grantButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Live camera preview */}
      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        {/* Corner reticle overlay */}
        <View style={styles.overlay}>
          <View style={styles.reticle}>
            <View style={[styles.corner, styles.topLeft]} />
            <View style={[styles.corner, styles.topRight]} />
            <View style={[styles.corner, styles.bottomLeft]} />
            <View style={[styles.corner, styles.bottomRight]} />
          </View>
          <Text style={styles.hint}>Frame the leaf inside the box</Text>
        </View>
      </CameraView>

      {/* Controls panel */}
      <ScrollView style={styles.controls} contentContainerStyle={styles.controlsContent}>
        <View style={styles.row}>
          {/* Crop stage */}
          <View style={styles.pickerWrap}>
            <Text style={styles.label}>Crop stage</Text>
            <View style={styles.pickerBox}>
              <Picker selectedValue={cropStage} onValueChange={setCropStage} style={styles.picker}>
                {CROP_STAGES.map(s => (
                  <Picker.Item key={s.value} label={s.label} value={s.value} />
                ))}
              </Picker>
            </View>
          </View>
          {/* Language */}
          <View style={styles.pickerWrap}>
            <Text style={styles.label}>Language</Text>
            <View style={styles.pickerBox}>
              <Picker selectedValue={language} onValueChange={setLanguage} style={styles.picker}>
                {LANGUAGES.map(l => (
                  <Picker.Item key={l.value} label={l.label} value={l.value} />
                ))}
              </Picker>
            </View>
          </View>
        </View>

        {loading ? (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#16a34a" />
            <Text style={styles.loadingText}>{statusMsg}</Text>
          </View>
        ) : (
          <TouchableOpacity style={styles.captureButton} onPress={handleCapture} activeOpacity={0.85}>
            <View style={styles.captureInner} />
          </TouchableOpacity>
        )}
      </ScrollView>
    </View>
  );
}

const CORNER = 24;
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },

  camera: { flex: 1 },

  overlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  reticle: {
    width: width * 0.65,
    height: width * 0.65,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: CORNER,
    height: CORNER,
    borderColor: '#fff',
    borderWidth: 3,
  },
  topLeft: { top: 0, left: 0, borderRightWidth: 0, borderBottomWidth: 0 },
  topRight: { top: 0, right: 0, borderLeftWidth: 0, borderBottomWidth: 0 },
  bottomLeft: { bottom: 0, left: 0, borderRightWidth: 0, borderTopWidth: 0 },
  bottomRight: { bottom: 0, right: 0, borderLeftWidth: 0, borderTopWidth: 0 },
  hint: { color: 'rgba(255,255,255,0.8)', marginTop: 12, fontSize: 13 },

  controls: { backgroundColor: '#fff', maxHeight: 220 },
  controlsContent: { padding: 16, gap: 12 },

  row: { flexDirection: 'row', gap: 12 },
  pickerWrap: { flex: 1 },
  label: { fontSize: 12, fontWeight: '600', color: '#6b7280', marginBottom: 4 },
  pickerBox: { borderWidth: 1, borderColor: '#d1d5db', borderRadius: 10, overflow: 'hidden' },
  picker: { height: 44 },

  loadingBox: { alignItems: 'center', paddingVertical: 12, gap: 8 },
  loadingText: { color: '#16a34a', fontWeight: '600', fontSize: 14 },

  captureButton: {
    alignSelf: 'center',
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#16a34a',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
  },
  captureInner: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: '#16a34a',
  },

  permissionBox: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32 },
  permissionEmoji: { fontSize: 56, marginBottom: 16 },
  permissionTitle: { fontSize: 20, fontWeight: '800', color: '#111827', textAlign: 'center' },
  permissionDesc: { fontSize: 14, color: '#6b7280', textAlign: 'center', marginTop: 8, lineHeight: 20 },
  grantButton: {
    marginTop: 24,
    backgroundColor: '#16a34a',
    borderRadius: 12,
    paddingHorizontal: 28,
    paddingVertical: 14,
  },
  grantButtonText: { color: '#fff', fontWeight: '700', fontSize: 16 },
});
