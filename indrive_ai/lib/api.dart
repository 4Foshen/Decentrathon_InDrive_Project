import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://10.0.2.2:8000";

  /// Статический метод для проверки состояния автомобиля
  static Future<String> checkCar(File imageFile) async {
    try {
      var uri = Uri.parse("$baseUrl/check_car/");
      var request = http.MultipartRequest('POST', uri);

      // Добавляем файл
      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
        ),
      );

      // Отправляем запрос
      var response = await request.send();

      if (response.statusCode == 200) {
        var respStr = await response.stream.bytesToString();
        var jsonResp = json.decode(respStr);
        return jsonResp['status']; // "clean" или "dirty"
      } else {
        throw Exception("Ошибка сервера: ${response.statusCode}");
      }
    } catch (e) {
      throw Exception("Ошибка отправки файла: $e");
    }
  }
}
