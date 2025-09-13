import 'dart:io';
import 'package:flutter/material.dart';
import 'package:indrive_ai/start_page.dart';
import 'driver_storage.dart';
import 'package:indrive_ai/theme/app_colors.dart';

class ResultPage extends StatefulWidget {
  final String driverName;
  final String carModel;
  final String carPhoto; // путь к файлу или ассету

  const ResultPage({
    super.key,
    required this.driverName,
    required this.carModel,
    required this.carPhoto,
  });

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  String condition = "Определяется...";

  @override
  void initState() {
    super.initState();
    _analyzeCar();
  }

  void _analyzeCar() async {
    // TODO: запрос в API для анализа состояния авто
    await Future.delayed(const Duration(seconds: 2));
    setState(() {
      condition = "Хорошее состояние"; // заглушка
    });
  }

  void _saveData() async {
    final driverData = {
      "name": widget.driverName,
      "car": widget.carModel,
      "photo": widget.carPhoto,
      "status": condition,
    };

    await DriverStorage.saveDriver(driverData);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Данные сохранены!")),
    );

    _goToMenu();
  }

  void _goToMenu() {
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (context) => const StartPage()),
      (route) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    final isFile = File(widget.carPhoto).existsSync(); // проверка: файл или нет

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text("Результат"),
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: AppColors.textPrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Фото авто
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: isFile
                  ? Image.file(File(widget.carPhoto),
                      height: 300, width: 300, fit: BoxFit.cover)
                  : Image.asset(widget.carPhoto,
                      height: 200, fit: BoxFit.cover),
            ),
            const SizedBox(height: 24),

            // Карточка с результатом
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 8,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Text(
                "Состояние авто: $condition",
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppColors.textPrimary,
                ),
              ),
            ),

            const SizedBox(height: 40),

            // Основная кнопка
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _saveData,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  elevation: 3,
                ),
                child: const Text(
                  "Сохранить и далее",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
