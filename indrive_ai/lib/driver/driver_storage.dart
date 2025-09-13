import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class DriverStorage {
  static const String _key = "drivers_list";

  /// Сохраняем нового водителя, добавляя к существующим
  static Future<void> saveDriver(Map<String, dynamic> driverData) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_key);

    List<Map<String, dynamic>> drivers = [];
    if (data != null) {
      drivers = List<Map<String, dynamic>>.from(jsonDecode(data));
    }

    drivers.add(driverData);

    await prefs.setString(_key, jsonEncode(drivers));
  }

  /// Загружаем всех сохранённых водителей
  static Future<List<Map<String, dynamic>>> loadDrivers() async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_key);

    if (data != null) {
      return List<Map<String, dynamic>>.from(jsonDecode(data));
    }
    return [];
  }

  /// Удаляем всех водителей
  static Future<void> clearDrivers() async {
    final prefs = await SharedPreferences.getInstance();
    prefs.remove(_key);
  }

  /// Удаляем одного водителя по индексу
  static Future<void> removeDriver(int index) async {
    final prefs = await SharedPreferences.getInstance();
    final data = prefs.getString(_key);

    if (data != null) {
      List<Map<String, dynamic>> drivers =
          List<Map<String, dynamic>>.from(jsonDecode(data));

      if (index >= 0 && index < drivers.length) {
        drivers.removeAt(index);
        await prefs.setString(_key, jsonEncode(drivers));
      }
    }
  }
}
