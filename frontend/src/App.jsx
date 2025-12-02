import React, { useState } from 'react';
import axios from 'axios';
import { Upload, CheckCircle, AlertCircle, Loader2, DollarSign, FileText, User } from 'lucide-react';
// import logo from './assets/logo.png'; // Descomenta cuando pongas tu logo

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const [formData, setFormData] = useState({
    nombre: '', direccion: '', correo: '', telefono: '', rfc: '',
    ingresoMensual: '', ingresoAnual: '', inversionMensual: '',
  });

  const [files, setFiles] = useState({
    pdfBuro: null, pdfDetallado: null, estadoCuenta: null, ine: null, comprobanteDomicilio: null
  });

  const handleInputChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });
  
  const handleFileChange = (e) => setFiles({ ...files, [e.target.name]: e.target.files[0] });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(''); setResult(null);

    const dataToSend = new FormData();
    Object.keys(formData).forEach(k => dataToSend.append(k, formData[k]));
    if (files.pdfBuro) dataToSend.append('pdfBuro', files.pdfBuro);
    if (files.pdfDetallado) dataToSend.append('pdfDetallado', files.pdfDetallado);
    if (files.estadoCuenta) dataToSend.append('estadoCuenta', files.estadoCuenta);
    if (files.ine) dataToSend.append('ine', files.ine);
    if (files.comprobanteDomicilio) dataToSend.append('comprobanteDomicilio', files.comprobanteDomicilio);

    try {
      // CONEXIÓN A PUERTO 8000
      const response = await axios.post('http://127.0.0.1:8000/predict', dataToSend, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError('Error al conectar. Verifica que main.py corra en puerto 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans text-gray-800">
      <nav className="bg-white shadow-md p-4 sticky top-0 z-50">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold text-credimed-primary">CREDI<span className="text-credimed-dark">MED</span></h1>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto p-6 mt-6">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-credimed-primary mb-2">Solicitud de Crédito</h1>
          <p className="text-gray-600">Evaluación financiera automatizada.</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8 border-t-4 border-credimed-primary">
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Sección 1 */}
            <div>
              <h2 className="text-xl font-semibold flex items-center gap-2 mb-4 text-credimed-dark"><User className="text-credimed-secondary"/> Datos Personales</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <InputGroup label="Nombre" name="nombre" value={formData.nombre} onChange={handleInputChange} required />
                <InputGroup label="RFC" name="rfc" value={formData.rfc} onChange={handleInputChange} required />
                <InputGroup label="Correo" name="correo" type="email" value={formData.correo} onChange={handleInputChange} required />
                <InputGroup label="Teléfono" name="telefono" value={formData.telefono} onChange={handleInputChange} required />
                <div className="md:col-span-2"><InputGroup label="Dirección" name="direccion" value={formData.direccion} onChange={handleInputChange} required /></div>
              </div>
            </div>
            <hr />
            {/* Sección 2 */}
            <div>
              <h2 className="text-xl font-semibold flex items-center gap-2 mb-4 text-credimed-dark"><DollarSign className="text-credimed-secondary"/> Finanzas (MXN)</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <InputGroup label="Ingreso Mensual" name="ingresoMensual" type="number" prefix="$" value={formData.ingresoMensual} onChange={handleInputChange} required />
                <InputGroup label="Ingreso Anual" name="ingresoAnual" type="number" prefix="$" value={formData.ingresoAnual} onChange={handleInputChange} required />
                <InputGroup label="Inversión Mensual" name="inversionMensual" type="number" prefix="$" value={formData.inversionMensual} onChange={handleInputChange} required />
              </div>
            </div>
            <hr />
            {/* Sección 3 - DOCUMENTOS */}
            <div>
              <h2 className="text-xl font-semibold flex items-center gap-2 mb-4 text-credimed-dark"><FileText className="text-credimed-secondary"/> Documentos</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FileInput label="Buró Score (PDF)" name="pdfBuro" onChange={handleFileChange} required fileName={files.pdfBuro?.name} />
                <FileInput label="Reporte Detallado (PDF)" name="pdfDetallado" onChange={handleFileChange} required fileName={files.pdfDetallado?.name} />
                <FileInput label="INE" name="ine" onChange={handleFileChange} required fileName={files.ine?.name} />
                <FileInput label="Comprobante Domicilio" name="comprobanteDomicilio" onChange={handleFileChange} required fileName={files.comprobanteDomicilio?.name} />
                <FileInput label="Estado Cuenta (Opcional)" name="estadoCuenta" onChange={handleFileChange} fileName={files.estadoCuenta?.name} />
              </div>
            </div>

            <button type="submit" disabled={loading} className="w-full bg-credimed-primary hover:bg-[#448a96] text-white font-bold py-4 rounded-lg flex justify-center items-center gap-2 text-lg shadow-md disabled:opacity-70">
              {loading ? <><Loader2 className="animate-spin"/> Procesando...</> : 'Calcular Crédito'}
            </button>
          </form>
        </div>

        {error && <div className="mt-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex gap-2"><AlertCircle/>{error}</div>}
        
        {result && (
          <div className="mt-8 bg-green-50 border border-green-200 rounded-xl p-8 text-center">
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h2 className="text-3xl font-bold text-gray-800">¡Aprobado!</h2>
            <p className="text-lg text-gray-600 mt-2">Línea sugerida:</p>
            <div className="text-5xl font-extrabold text-credimed-primary mt-4">
              ${Number(result.monto).toLocaleString('es-MX')} MXN
            </div>
            <p className="mt-4 text-sm text-gray-500">Score Predicho: {result.credit_score_predicho}</p>
          </div>
        )}
      </div>
    </div>
  );
}

const InputGroup = ({ label, name, type="text", value, onChange, required, prefix }) => (
  <div className="flex flex-col gap-1">
    <label className="text-sm font-medium text-gray-700">{label} {required && '*'}</label>
    <div className="relative">
      {prefix && <span className="absolute left-3 top-2.5 text-gray-400">{prefix}</span>}
      <input type={type} name={name} value={value} onChange={onChange} required={required} className={`w-full border border-gray-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-credimed-primary ${prefix ? 'pl-7' : ''}`} />
    </div>
  </div>
);

const FileInput = ({ label, name, onChange, required, fileName }) => (
  <div className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer relative group ${fileName ? 'border-credimed-primary bg-blue-50' : 'border-gray-300 hover:bg-gray-50'}`}>
    <input type="file" name={name} onChange={onChange} required={required && !fileName} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
    <div className="flex flex-col items-center gap-2">
      {fileName ? <><CheckCircle className="text-credimed-primary w-8 h-8"/><span className="text-sm font-bold text-credimed-primary break-all">{fileName}</span></> : <><Upload className="text-gray-400 w-6 h-6"/><span className="text-sm text-gray-600">{label} {required && '*'}</span></>}
    </div>
  </div>
);

export default App;
